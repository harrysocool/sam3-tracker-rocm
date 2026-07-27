// SAM3 ViT backbone C++ host (504px), guarded Phase 2 dynamic-K v5 candidate.
// Window QKV runs before partition and O projection after unpartition, so all
// projection GEMMs use M=1536 while preserving padded-token attention semantics.
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/experimental/xrt_xclbin.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <immintrin.h>
#include <omp.h>
using std::vector; using std::string;
typedef uint16_t bf16;
static inline bf16 f2b(float f){ uint32_t x; std::memcpy(&x,&f,4); uint32_t r=x+0x7fff+((x>>16)&1); return (bf16)(r>>16); }
static inline float b2f(bf16 h){ uint32_t x=((uint32_t)h)<<16; float f; std::memcpy(&f,&x,4); return f; }
static void f2b_bulk(bf16* d,const float* s,size_t n){ size_t i=0; for(;i+16<=n;i+=16){ __m512 v=_mm512_loadu_ps(s+i); __m256bh b=_mm512_cvtneps_pbh(v); _mm256_storeu_si256((__m256i*)(d+i),(__m256i)b);} for(;i<n;i++)d[i]=f2b(s[i]); }

static const int C=1024,d=64,nH=16,Hid=4736,Hgemm=4864,Hpad=5120,Nhalf=2560,MFFN=1536,GRID=36,S_G=1296,Sp_ln=1344;
static xrt::device DEV;
const string CBB="/home/amd/project/npu_iron/weights/cbb/";

vector<float> loadf(const string&p){ std::ifstream f(p+".bin",std::ios::binary|std::ios::ate); size_t n=f.tellg()/4; f.seekg(0); vector<float> v(n); f.read((char*)v.data(),n*4); return v; }

struct H { xrt::kernel k; xrt::bo bi; uint32_t nw; };
H loadx(const string&dir){
  auto xclb=xrt::xclbin(dir+"/final.xclbin");
  auto uuid=DEV.register_xclbin(xclb);
  auto ctx=xrt::hw_context(DEV,uuid);
  auto k=xrt::kernel(ctx,"MLIR_AIE");
  std::ifstream f(dir+"/insts.bin",std::ios::binary|std::ios::ate); size_t nb=f.tellg(); f.seekg(0);
  vector<uint8_t> ib(nb); f.read((char*)ib.data(),nb);
  auto bi=xrt::bo(DEV,nb,xrt::bo::flags::cacheable,k.group_id(1));
  bi.write(ib.data()); bi.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  return {k,bi,(uint32_t)(nb/4)};
}
H loadi(const H&base,const string&dir){
  std::ifstream f(dir+"/insts.bin",std::ios::binary|std::ios::ate); size_t nb=f.tellg(); f.seekg(0);
  vector<uint8_t> ib(nb); f.read((char*)ib.data(),nb);
  auto bi=xrt::bo(DEV,nb,xrt::bo::flags::cacheable,base.k.group_id(1));
  bi.write(ib.data()); bi.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  return {base.k,bi,(uint32_t)(nb/4)};
}
xrt::bo mkbo(H&h,int gid,size_t bytes){ return xrt::bo(DEV,bytes,xrt::bo::flags::host_only,h.k.group_id(gid)); }
static vector<bf16> _wbuf;
void wbf(xrt::bo&bo,const float*src,size_t n){ if(_wbuf.size()<n)_wbuf.resize(n); f2b_bulk(_wbuf.data(),src,n); bo.write(_wbuf.data(),(size_t)n*2,0); bo.sync(XCL_BO_SYNC_BO_TO_DEVICE); }
void wbf_v(xrt::bo&bo,const vector<float>&v){ wbf(bo,v.data(),v.size()); }
void wbf_mapped(xrt::bo&bo,const float*src,size_t valid,size_t total,bool&initialized){
  bf16*dst=bo.map<bf16*>();
  if(!initialized){ std::memset(dst,0,total*sizeof(bf16)); initialized=true; }
  f2b_bulk(dst,src,valid);
  bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
}
void rdf(xrt::bo&bo,float*dst,size_t n){ bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE); bo.read(dst); }
void rdbf(xrt::bo&bo,float*dst,size_t n){ bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE); vector<bf16> t(n); bo.read(t.data()); for(size_t i=0;i<n;i++)dst[i]=b2f(t[i]); }

double T_disp=0,T_host=0;
static inline double now(){ return std::chrono::duration<double,std::milli>(std::chrono::high_resolution_clock::now().time_since_epoch()).count(); }
struct Profile {
  double res_copy=0,ln1=0,partition=0,qkv_pack=0,qkv_disp=0,qkv_read=0,qkv_split=0;
  double attn_pack=0,attn_h2d=0,flash_disp=0,attn_d2h=0,attn_unpack=0;
  double attn_layout=0,opack=0,odisp=0,oread=0,unpartition_res1=0;
  double res2_copy=0,ln2=0,ffn1_pack=0,ffn1_disp=0,ffn1_sync=0;
  double gelu_pack=0,ffn2_disp=0,ffn2_sync_res=0;
};
Profile P;
int CUR_LAYER=-1;
#define DISP_ACC(call,acc,name) do{ double _t=now(); (call).wait(); double _dt=now()-_t; T_disp+=_dt; (acc)+=_dt; if(_dt>20.0) fprintf(stderr,"slow_dispatch layer=%d stage=%s ms=%.3f\n",CUR_LAYER,name,_dt); }while(0)
void print_profile(FILE*fp){
  fprintf(fp,"profile res_copy=%.3f ln1=%.3f partition=%.3f qkv_pack=%.3f qkv_disp=%.3f qkv_read=%.3f qkv_split=%.3f "
    "attn_pack=%.3f attn_h2d=%.3f flash_disp=%.3f attn_d2h=%.3f attn_unpack=%.3f attn_layout=%.3f "
    "opack=%.3f odisp=%.3f oread=%.3f unpartition_res1=%.3f res2_copy=%.3f ln2=%.3f "
    "ffn1_pack=%.3f ffn1_disp=%.3f ffn1_sync=%.3f gelu_pack=%.3f ffn2_disp=%.3f ffn2_sync_res=%.3f\n",
    P.res_copy,P.ln1,P.partition,P.qkv_pack,P.qkv_disp,P.qkv_read,P.qkv_split,
    P.attn_pack,P.attn_h2d,P.flash_disp,P.attn_d2h,P.attn_unpack,P.attn_layout,
    P.opack,P.odisp,P.oread,P.unpartition_res1,P.res2_copy,P.ln2,
    P.ffn1_pack,P.ffn1_disp,P.ffn1_sync,P.gelu_pack,P.ffn2_disp,P.ffn2_sync_res);
}

static inline double nw3(){return 0;}
// reused scratch: RZ=resize(no zero, fully overwritten), ZB=zero whole (padded)
#define RZv(v,n) static vector<float> v; v.resize(n);
#define RZb(v,n) static vector<bf16> v; v.resize(n);
int GLOBAL[4]={7,15,23,31};
bool isglob(int li){ for(int g:GLOBAL) if(li==g) return true; return false; }

// global handles
H hln,hqkv_w,hqkv_g,ho_w,ho_g,hqt_w,hsm_w,hpv_w,hgelu,hf2,hf1v2,hflashw,hflashg;
// LN bos
xrt::bo lin,lout;
// proj scratch
xrt::bo qkv_wA,qkv_wC,qkv_gA,qkv_gC,o_wA,o_wC,o_gA,o_gC;
// attn bos (win/glob): qA,qB,SC(shared sm-in),P(shared pv-in),pB,pC
xrt::bo qaW,qbW,scW,pW,pbW,pcW; long NBW;
xrt::bo qaF,qbF,pbF,OF;
xrt::bo qaG,qbG,scG,pG,pbG,pcG,OFg; long NBG;
// ffn
xrt::bo f1A,f1C,gin,gsh,f2C,f1Av2,f1Cfull;
// resident weights per layer
vector<xrt::bo> WB_qkv(32),WB_o(32),WB_w1a(32),WB_w1b(32),WB_w2(32),WB_w1full(32);
// cpu-side weights (bias/ln)
vector<vector<float>> bqkv(32),Ob(32),ln1w(32),ln1b(32),ln2w(32),ln2b(32),b1(32),fc2b(32);
vector<float> ropeWc,ropeWs,ropeGc,ropeGs;

void npu_ln(const vector<float>&x,const vector<float>&w,const vector<float>&b,vector<float>&out){
  // CPU LayerNorm (all f32) — replaces NPU LN dispatch (saves 64 dispatches ~218ms). More accurate than bf16-intermediate NPU LN.
  int S=x.size()/C; out.resize(S*C);
  #pragma omp parallel for schedule(static)
  for(int s=0;s<S;s++){ const float*xr=&x[s*C]; float*orow=&out[s*C];
    float mean=0.f; for(int cc=0;cc<C;cc++) mean+=xr[cc]; mean/=C;
    float var=0.f; for(int cc=0;cc<C;cc++){ float dd=xr[cc]-mean; var+=dd*dd; } var/=C;
    float inv=1.f/std::sqrt(var+1e-6f);
    for(int cc=0;cc<C;cc++) orow[cc]=(xr[cc]-mean)*inv*w[cc]+b[cc]; }
}
// rope on [G,S,64]: q*cos + rotate_pairwise(q)*sin
void rope(vector<float>&q,int G,int S,const vector<float>&cs,const vector<float>&sn){
  #pragma omp parallel for collapse(2) schedule(static)
  for(int g=0;g<G;g++)for(int s=0;s<S;s++){ float*row=&q[(g*S+s)*d]; const float*co=&cs[s*d]; const float*si=&sn[s*d];
    for(int i=0;i<d;i+=2){ float a=row[i],b=row[i+1]; row[i]=a*co[i]-b*si[i]; row[i+1]=b*co[i+1]+a*si[i+1]; } }
}
// gelu erf-poly
static inline float gelu1(float x){ float a=x*0.7071067811865476f; float t=1.f/(1.f+0.3275911f*std::fabs(a));
  float p=t*(0.254829592f+t*(-0.284496736f+t*(1.421413741f+t*(-1.453152027f+t*1.061405429f))));
  float e=(1.f-p*std::exp(-a*a))*(a<0?-1.f:1.f); return 0.5f*x*(1.f+e); }

// Build Q/K/V directly from mapped FP32 projection output into mapped BF16 flash BOs.
void attn(const float*qkv,const vector<float>&bq,bool glob,xrt::bo&oA){
  int G=glob?16:64, S=glob?1296:576, Sp=glob?1344:576; long NB=glob?NBG:NBW;
  if(!glob){ // window flash attention: fused qkt+softmax+pv, 1 dispatch, Q unscaled (flash folds 1/sqrt(d))
    double tp=now();
    bf16*Qf=qaF.map<bf16*>(), *Kf=qbF.map<bf16*>(), *Vf=pbF.map<bf16*>();
    static const vector<float> zero_qkv(3*C,0.f);
    { // Split+bias+RoPE+BF16 in one pass directly into the flash input BOs.
      const float*co=ropeWc.data(), *si=ropeWs.data();
      _Pragma("omp parallel for schedule(static)") for(int t=0;t<2304;t++){
        int win=t/576,p=t%576;
        int w0=win/2,w1=win%2,i=p/24,j=p%24;
        int gi=w0*24+i,gj=w1*24+j;
        const float*qrow=(gi<GRID&&gj<GRID)?qkv+((size_t)gi*GRID+gj)*3*C:zero_qkv.data();
        const float*cos_s=co+p*d, *sin_s=si+p*d;
        for(int h=0;h<nH;h++){
          int g=win*nH+h; size_t out=((size_t)g*S+p)*d;
        for(int i=0;i<d;i+=2){
          float qi=qrow[h*d+i]+bq[h*d+i], qi1=qrow[h*d+i+1]+bq[h*d+i+1];
          float ki=qrow[C+h*d+i]+bq[C+h*d+i], ki1=qrow[C+h*d+i+1]+bq[C+h*d+i+1];
          Qf[out+i]  =f2b(qi*cos_s[i]  -qi1*sin_s[i]);
          Qf[out+i+1]=f2b(qi1*cos_s[i+1]+qi*sin_s[i+1]);
          Kf[out+i]  =f2b(ki*cos_s[i]  -ki1*sin_s[i]);
          Kf[out+i+1]=f2b(ki1*cos_s[i+1]+ki*sin_s[i+1]); }
          for(int c=0;c<d;c++) Vf[out+c]=f2b(qrow[2*C+h*d+c]+bq[2*C+h*d+c]);
        }
      }
    }
    P.attn_pack+=now()-tp; tp=now();
    qaF.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    qbF.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    pbF.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    P.attn_h2d+=now()-tp;
    DISP_ACC(hflashw.k(3,hflashw.bi,hflashw.nw,qaF,qbF,pbF,OF),P.flash_disp,"flash_w");
    tp=now();
    OF.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    P.attn_d2h+=now()-tp; tp=now();
    const bf16*src=OF.map<const bf16*>(); bf16*dst=oA.map<bf16*>();
    std::memset(dst,0,(size_t)MFFN*C*sizeof(bf16));
    _Pragma("omp parallel for schedule(static)") for(int t=0;t<S_G;t++){
      int gi=t/GRID,gj=t%GRID,win=(gi/24)*2+(gj/24),p=(gi%24)*24+(gj%24);
      for(int h=0;h<nH;h++)for(int c=0;c<d;c++)
        dst[(size_t)t*C+h*d+c]=src[((size_t)(win*nH+h)*576+p)*d+c];
    }
    oA.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    P.attn_layout+=now()-tp;
    return;
  }
  // Global: flash_g (1 dispatch). RoPE applied inline during BF16 packing.
  // flash_g applies 1/sqrt(d) internally. Zero-padded rows handle masking.
  double tp=now();
  bf16*Q=qaG.map<bf16*>(), *K=qbG.map<bf16*>(), *V=pbG.map<bf16*>();
  std::memset(Q,0,(size_t)G*Sp*d*sizeof(bf16)); std::memset(K,0,(size_t)G*Sp*d*sizeof(bf16)); std::memset(V,0,(size_t)G*Sp*d*sizeof(bf16));
  { const float*co=ropeGc.data(), *si=ropeGs.data();
    #pragma omp parallel for schedule(static)
    for(int t=0;t<S;t++){
      const float*qrow=qkv+t*3072; const float*cos_s=co+t*d, *sin_s=si+t*d;
      for(int h=0;h<nH;h++){ size_t out=((size_t)h*Sp+t)*d;
      for(int i=0;i<d;i+=2){
        float qi=qrow[h*d+i]+bq[h*d+i], qi1=qrow[h*d+i+1]+bq[h*d+i+1];
        float ki=qrow[C+h*d+i]+bq[C+h*d+i], ki1=qrow[C+h*d+i+1]+bq[C+h*d+i+1];
        Q[out+i]  =f2b(qi*cos_s[i]  -qi1*sin_s[i]);
        Q[out+i+1]=f2b(qi1*cos_s[i+1]+qi*sin_s[i+1]);
        K[out+i]  =f2b(ki*cos_s[i]  -ki1*sin_s[i]);
        K[out+i+1]=f2b(ki1*cos_s[i+1]+ki*sin_s[i+1]); }
        for(int c=0;c<d;c++) V[out+c]=f2b(qrow[2*C+h*d+c]+bq[2*C+h*d+c]);
      }
    }
  }
  P.attn_pack+=now()-tp; tp=now();
  qaG.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  qbG.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  pbG.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  P.attn_h2d+=now()-tp;
  DISP_ACC(hflashg.k(3,hflashg.bi,hflashg.nw,qaG,qbG,pbG,OFg),P.flash_disp,"flash_g");  // flash global → bf16
  tp=now();
  OFg.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  P.attn_d2h+=now()-tp; tp=now();
  const bf16*src=OFg.map<const bf16*>(); bf16*dst=oA.map<bf16*>();
  std::memset(dst,0,(size_t)1536*C*sizeof(bf16));
  _Pragma("omp parallel for schedule(static)") for(int t=0;t<1296;t++)for(int h=0;h<nH;h++)for(int c=0;c<d;c++) dst[t*C+h*d+c]=src[(h*Sp+t)*d+c];
  oA.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  P.attn_layout+=now()-tp;
}

// one block: x [1296,C] -> [1296,C]
void block(vector<float>&x,int li){
  CUR_LAYER=li;
  bool glob=isglob(li);
  double tp=now(); static vector<float> res; res=x; P.res_copy+=now()-tp;
  tp=now(); vector<float> xn; npu_ln(x,ln1w[li],ln1b[li],xn); P.ln1+=now()-tp;  // [1296,C]

  // QKV projection is token-wise: run it on the 1296 valid tokens before
  // window partition. The M=1536 global instruction stream serves all layers.
  H&hq=hqkv_g; xrt::bo&qA=qkv_gA; xrt::bo&qCo=qkv_gC;
  tp=now(); static bool qkv_input_initialized=false;
  wbf_mapped(qA,xn.data(),(size_t)S_G*C,(size_t)MFFN*C,qkv_input_initialized);
  P.qkv_pack+=now()-tp;
  DISP_ACC(hq.k(3,hq.bi,hq.nw,qA,WB_qkv[li],qCo),P.qkv_disp,"qkv_m1536");
  tp=now(); qCo.sync(XCL_BO_SYNC_BO_FROM_DEVICE); const float*qkv=qCo.map<const float*>(); P.qkv_read+=now()-tp;
  const vector<float>&bq=bqkv[li];
  H&ho=ho_g; xrt::bo&oA=o_gA; xrt::bo&oCo=o_gC;
  // Split, bias, RoPE, and both flash BO layouts are produced directly from mapped QKV output.
  attn(qkv,bq,glob,oA);
  // Window attention has already been gathered back to token-major valid rows.
  // O projection therefore also uses the common M=1536 instruction stream.
  DISP_ACC(ho.k(3,ho.bi,ho.nw,oA,WB_o[li],oCo),P.odisp,"oproj_m1536");
  tp=now(); oCo.sync(XCL_BO_SYNC_BO_FROM_DEVICE); const float*ao=oCo.map<const float*>(); P.oread+=now()-tp;
  // +Ob
  const vector<float>&ob=Ob[li];
  // Both window and global O projection outputs are token-major [1296,C].
  tp=now(); static vector<float> attn_out; attn_out.assign(1296*C,0.f);
  for(int t=0;t<1296;t++)for(int c=0;c<C;c++) attn_out[t*C+c]=ao[t*C+c]+ob[c];

  // residual1
  #pragma omp parallel for schedule(static)
  for(int i=0;i<1296*C;i++) x[i]=res[i]+attn_out[i];
  P.unpartition_res1+=now()-tp;

  // ffn
  tp=now(); static vector<float> res2; res2=x; P.res2_copy+=now()-tp;
  tp=now(); vector<float> xn2; npu_ln(x,ln2w[li],ln2b[li],xn2); P.ln2+=now()-tp;
  tp=now(); static bool ffn1_input_initialized=false;
  wbf_mapped(f1Av2,xn2.data(),(size_t)S_G*C,(size_t)MFFN*C,ffn1_input_initialized);
  P.ffn1_pack+=now()-tp;
  DISP_ACC(hf1v2.k(3,hf1v2.bi,hf1v2.nw,f1Av2,WB_w1full[li],f1Cfull),P.ffn1_disp,"ffn1");
  tp=now(); f1Cfull.sync(XCL_BO_SYNC_BO_FROM_DEVICE); P.ffn1_sync+=now()-tp;
  const float*hid_=f1Cfull.map<float*>(); // zero-copy: direct BO pointer
  // +b1 + gelu -> gin bf16
  tp=now(); { // Padded W1/bias/W2 regions are exactly zero; only valid GELU values contribute.
    static bool gelu_output_initialized=false;
    bf16*gb=gsh.map<bf16*>();
    if(!gelu_output_initialized){ std::memset(gb,0,(size_t)MFFN*Hpad*sizeof(bf16)); gelu_output_initialized=true; }
    const float*br=b1[li].data();
    _Pragma("omp parallel for schedule(static)") for(int r=0;r<S_G;r++){ const float*hr=hid_+(size_t)r*Hgemm; bf16*ob=gb+(size_t)r*Hpad;
      for(int cc=0;cc<Hid;cc++) ob[cc]=f2b(gelu1(hr[cc]+br[cc])); }
    gsh.sync(XCL_BO_SYNC_BO_TO_DEVICE); }
  P.gelu_pack+=now()-tp;
  // hgelu dispatch skipped (CPU gelu faster)
  DISP_ACC(hf2.k(3,hf2.bi,hf2.nw,gsh,WB_w2[li],f2C),P.ffn2_disp,"ffn2");
  tp=now(); f2C.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  const float*fo_=reinterpret_cast<const float*>(f2C.map<float*>());
  const vector<float>&f2b_=fc2b[li];
  _Pragma("omp parallel for schedule(static)") for(int r=0;r<1296;r++){ const float*rr=&res2[r*C]; const float*fr=fo_+r*C; float*xr=&x[r*C]; int cc=0;
    for(;cc+16<=C;cc+=16){ __m512 v=_mm512_add_ps(_mm512_add_ps(_mm512_loadu_ps(rr+cc),_mm512_loadu_ps(fr+cc)),_mm512_loadu_ps(f2b_.data()+cc)); _mm512_storeu_ps(xr+cc,v);}
    for(;cc<C;cc++) xr[cc]=rr[cc]+fr[cc]+f2b_[cc]; }
  P.ffn2_sync_res+=now()-tp;
}

static string input_file="";
static string output_file="";
static string microbench="";

int main(int argc,char**argv){
  const char*allow_wip=std::getenv("ALLOW_DYNAMIC_K_V5_WIP");
  if(!allow_wip||string(allow_wip)!="1"){
    fprintf(stderr,"refusing to run unvalidated dynamic-K v5 backbone; set ALLOW_DYNAMIC_K_V5_WIP=1 only after ABI gate\n");
    return 2;
  }
  int n_runs=3;
  int micro_iters=2000;
  for(int i=1;i<argc;i++){
    string a=argv[i];
    if(a=="--input"&&i+1<argc)  { input_file=argv[++i]; }
    else if(a=="--output"&&i+1<argc){ output_file=argv[++i]; }
    else if(a=="--runs"&&i+1<argc)  { n_runs=atoi(argv[++i]); }
    else if(a=="--microbench"&&i+1<argc){ microbench=argv[++i]; }
    else if(a=="--micro-iters"&&i+1<argc){ micro_iters=atoi(argv[++i]); }
  }
  DEV=xrt::device(0);
  const string A="/home/amd/project/npu_iron/sam3_attn/";
  const string S=A+"shared_gemm_dynamic_rtp_v5/";
  const string CF=A+"compact_ffn_dynamic_rtp_v5/";
  hqkv_w=loadx(S+"qkv_w");
  hqkv_g=loadi(hqkv_w,S+"qkv_g");
  ho_w=loadi(hqkv_w,S+"o_w");
  ho_g=loadi(hqkv_w,S+"o_g");
  hf1v2=loadi(hqkv_w,CF+"ffn1");
  hf2=loadi(hqkv_w,S+"ffn2");
  hflashw=loadx(A+"attn_v2/flash_w"); hflashg=loadx(A+"attn_v2/flash_g");
  fprintf(stderr,"xclbins loaded\n");
  qkv_wA=mkbo(hqkv_w,3,(size_t)2304*C*2); qkv_wC=mkbo(hqkv_w,5,(size_t)2304*3072*4);
  qkv_gA=mkbo(hqkv_g,3,(size_t)1536*C*2); qkv_gC=mkbo(hqkv_g,5,(size_t)1536*3072*4);
  o_wA=mkbo(ho_w,3,(size_t)2304*C*2); o_wC=mkbo(ho_w,5,(size_t)2304*C*4);
  o_gA=mkbo(ho_g,3,(size_t)1536*C*2); o_gC=mkbo(ho_g,5,(size_t)1536*C*4);
  NBW=(long)64*576*576; NBG=(long)16*1344*1344;
  qaF=mkbo(hflashw,3,(size_t)64*576*d*2); qbF=mkbo(hflashw,4,(size_t)64*576*d*2); pbF=mkbo(hflashw,5,(size_t)64*576*d*2); OF=mkbo(hflashw,6,(size_t)64*576*d*2);
  qaG=mkbo(hflashg,3,(size_t)16*1344*d*2); qbG=mkbo(hflashg,4,(size_t)16*1344*d*2);
  pbG=mkbo(hflashg,5,(size_t)16*1344*d*2); OFg=mkbo(hflashg,6,(size_t)16*1344*d*2);
  f1Av2=mkbo(hf1v2,3,(size_t)MFFN*C*2); f1Cfull=mkbo(hf1v2,5,(size_t)MFFN*Hgemm*4);
  gsh=mkbo(hf2,3,(size_t)MFFN*Hpad*2); f2C=mkbo(hf2,5,(size_t)MFFN*C*4);
  fprintf(stderr,"bos allocated\n");
  // load weights resident
  for(int li=0;li<32;li++){ bool glob=isglob(li); char b[64];
    auto Wq=loadf(CBB+"L"+std::to_string(li)+"_Wqkv"); WB_qkv[li]=mkbo(glob?hqkv_g:hqkv_w,4,(size_t)C*3072*2); wbf_v(WB_qkv[li],Wq);
    auto Wo=loadf(CBB+"L"+std::to_string(li)+"_Ow"); WB_o[li]=mkbo(glob?ho_g:ho_w,4,(size_t)C*C*2); wbf_v(WB_o[li],Wo);
    auto W1=loadf(CBB+"L"+std::to_string(li)+"_W1"); // [C,Hpad]
    vector<float>W1compact((size_t)C*Hgemm);
    for(int r=0;r<C;r++) std::memcpy(W1compact.data()+(size_t)r*Hgemm,W1.data()+(size_t)r*Hpad,(size_t)Hgemm*4);
    WB_w1full[li]=mkbo(hf1v2,4,(size_t)C*Hgemm*2); wbf_v(WB_w1full[li],W1compact);
    auto W2=loadf(CBB+"L"+std::to_string(li)+"_W2"); WB_w2[li]=mkbo(hf2,4,(size_t)Hpad*C*2); wbf_v(WB_w2[li],W2);
    bqkv[li]=loadf(CBB+"L"+std::to_string(li)+"_bqkv"); Ob[li]=loadf(CBB+"L"+std::to_string(li)+"_Ob");
    ln1w[li]=loadf(CBB+"L"+std::to_string(li)+"_ln1w"); ln1b[li]=loadf(CBB+"L"+std::to_string(li)+"_ln1b");
    ln2w[li]=loadf(CBB+"L"+std::to_string(li)+"_ln2w"); ln2b[li]=loadf(CBB+"L"+std::to_string(li)+"_ln2b");
    b1[li]=loadf(CBB+"L"+std::to_string(li)+"_b1"); fc2b[li]=loadf(CBB+"L"+std::to_string(li)+"_fc2b");
  }
  ropeWc=loadf(CBB+"rope_win_cos"); ropeWs=loadf(CBB+"rope_win_sin");
  ropeGc=loadf(CBB+"rope_glob_cos"); ropeGs=loadf(CBB+"rope_glob_sin");
  fprintf(stderr,"weights resident\n");

  if(!microbench.empty()){
    auto zero_bo=[](xrt::bo&bo,size_t n){ vector<bf16> z(n,0); bo.write(z.data(),n*2,0); bo.sync(XCL_BO_SYNC_BO_TO_DEVICE); };
    zero_bo(qkv_wA,(size_t)2304*C); zero_bo(o_wA,(size_t)2304*C);
    zero_bo(f1Av2,(size_t)MFFN*C); zero_bo(gsh,(size_t)MFFN*Hpad);
    zero_bo(qaF,(size_t)64*576*d); zero_bo(qbF,(size_t)64*576*d); zero_bo(pbF,(size_t)64*576*d);
    vector<double> times; times.reserve((size_t)micro_iters*5); long ordinal=0;
    auto measure=[&](const char*name,auto launch){ double t=now(); launch().wait(); double dt=now()-t; times.push_back(dt); if(dt>20.0) printf("micro_slow ordinal=%ld stage=%s ms=%.3f\n",ordinal,name,dt); ordinal++; };
    for(int i=0;i<micro_iters;i++){
      if(microbench=="same-oproj"){
        measure("oproj_w",[&](){ return ho_w.k(3,ho_w.bi,ho_w.nw,o_wA,WB_o[0],o_wC); });
      } else if(microbench=="same-qkv"){
        measure("qkv_w",[&](){ return hqkv_w.k(3,hqkv_w.bi,hqkv_w.nw,qkv_wA,WB_qkv[0],qkv_wC); });
      } else if(microbench=="cycle"){
        measure("qkv_w",[&](){ return hqkv_w.k(3,hqkv_w.bi,hqkv_w.nw,qkv_wA,WB_qkv[0],qkv_wC); });
        measure("flash_w",[&](){ return hflashw.k(3,hflashw.bi,hflashw.nw,qaF,qbF,pbF,OF); });
        measure("oproj_w",[&](){ return ho_w.k(3,ho_w.bi,ho_w.nw,o_wA,WB_o[0],o_wC); });
        measure("ffn1",[&](){ return hf1v2.k(3,hf1v2.bi,hf1v2.nw,f1Av2,WB_w1full[0],f1Cfull); });
        measure("ffn2",[&](){ return hf2.k(3,hf2.bi,hf2.nw,gsh,WB_w2[0],f2C); });
      } else {
        fprintf(stderr,"unknown microbench: %s\n",microbench.c_str()); return 2;
      }
    }
    std::sort(times.begin(),times.end());
    auto quant=[&](double q){ double x=(times.size()-1)*q; size_t lo=(size_t)x,hi=std::min(lo+1,times.size()-1); return times[lo]+(times[hi]-times[lo])*(x-lo); };
    printf("micro_summary mode=%s calls=%zu p50=%.3f p95=%.3f max=%.3f\n",microbench.c_str(),times.size(),quant(.5),quant(.95),times.back());
    return 0;
  }
  // ── Persistent server mode (default) ────────────────────────────────
  // Protocol (binary, fixed sizes):
  //   Python → stdin:  int32 magic(0xBF16), float32[S*C] tokens
  //   C++    → stdout: int32 magic(0xBF16), float32[S*C] features
  // Stays alive until stdin closes (Python subprocess exits).
  // Weight loading happens once at startup → each inference ~2.3s not ~8s.
  if(input_file.empty()){
    const int S=S_G, N=S*C;           // 1296 * 1024
    const int MAGIC=0x0000BF16;
    vector<float> x(N);

    // Signal ready
    fwrite(&MAGIC, 4, 1, stdout); fflush(stdout);

    while(true){
      // Read magic + tokens from stdin
      int magic=0;
      if(fread(&magic,4,1,stdin)!=1) break;   // EOF → Python exited
      if(magic!=MAGIC){ fprintf(stderr,"bad magic %x\n",magic); break; }
      if((int)fread(x.data(),4,N,stdin)!=N) break;

      // Run inference
      T_disp=0; P={}; double t0=now();
      for(int li=0;li<32;li++) block(x,li);
      double wall=now()-t0;
      fprintf(stderr,"wall=%.0fms dispatch=%.0fms\n",wall,T_disp); fflush(stderr);
      print_profile(stderr); fflush(stderr);

      // Write magic + features to stdout
      fwrite(&MAGIC,4,1,stdout);
      fwrite(x.data(),4,N,stdout);
      fflush(stdout);
    }
    return 0;
  }

  // ── One-shot mode (--input/--output for testing) ───────────────────
  vector<float> x0;
  {
    std::ifstream fin(input_file,std::ios::binary|std::ios::ate);
    size_t n=fin.tellg()/4; fin.seekg(0); x0.resize(n); fin.read((char*)x0.data(),n*4);
  }
  auto ref=loadf(CBB+"final_feat");
  vector<float> x=x0;
  for(int li=0;li<32;li++) block(x,li);
  double dot=0,na=0,nb=0; for(size_t i=0;i<x.size();i++){ dot+=x[i]*ref[i]; na+=x[i]*x[i]; nb+=ref[i]*ref[i]; }
  printf("cos vs PyTorch = %.5f\n", dot/(std::sqrt(na)*std::sqrt(nb)+1e-9));
  for(int r=0;r<n_runs;r++){ T_disp=0; P={}; double t0=now(); x=x0; for(int li=0;li<32;li++) block(x,li); double wall=now()-t0;
    printf("run%d: wall=%.0fms (%.2f FPS)  dispatch=%.0fms\n",r,wall,1000.0/wall,T_disp);
    print_profile(stdout);
    if(r==0&&!output_file.empty()){
      FILE*fp=fopen(output_file.c_str(),"wb");
      if(fp){ fwrite(x.data(),4,x.size(),fp); fclose(fp); }
    }
  }
  return 0;
}

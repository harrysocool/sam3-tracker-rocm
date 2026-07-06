// SAM3 ViT backbone C++ host (504px). All element-wise on CPU; matmul/softmax/gelu/LN on NPU via XRT.
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/experimental/xrt_xclbin.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <immintrin.h>
#include <omp.h>
using std::vector; using std::string;
typedef uint16_t bf16;
static inline bf16 f2b(float f){ uint32_t x; std::memcpy(&x,&f,4); uint32_t r=x+0x7fff+((x>>16)&1); return (bf16)(r>>16); }
static inline float b2f(bf16 h){ uint32_t x=((uint32_t)h)<<16; float f; std::memcpy(&f,&x,4); return f; }
static void f2b_bulk(bf16* d,const float* s,size_t n){ size_t i=0; for(;i+16<=n;i+=16){ __m512 v=_mm512_loadu_ps(s+i); __m256bh b=_mm512_cvtneps_pbh(v); _mm256_storeu_si256((__m256i*)(d+i),(__m256i)b);} for(;i<n;i++)d[i]=f2b(s[i]); }

static const int C=1024,d=64,nH=16,Hid=4736,Hpad=5120,Nhalf=2560,MFFN=1536,GRID=36,S_G=1296,Sp_ln=1344;
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
xrt::bo mkbo(H&h,int gid,size_t bytes){ return xrt::bo(DEV,bytes,xrt::bo::flags::host_only,h.k.group_id(gid)); }
static vector<bf16> _wbuf;
void wbf(xrt::bo&bo,const float*src,size_t n){ if(_wbuf.size()<n)_wbuf.resize(n); f2b_bulk(_wbuf.data(),src,n); bo.write(_wbuf.data(),(size_t)n*2,0); bo.sync(XCL_BO_SYNC_BO_TO_DEVICE); }
void wbf_v(xrt::bo&bo,const vector<float>&v){ wbf(bo,v.data(),v.size()); }
void rdf(xrt::bo&bo,float*dst,size_t n){ bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE); bo.read(dst); }
void rdbf(xrt::bo&bo,float*dst,size_t n){ bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE); vector<bf16> t(n); bo.read(t.data()); for(size_t i=0;i<n;i++)dst[i]=b2f(t[i]); }

double T_disp=0,T_host=0;
static inline double now(){ return std::chrono::duration<double,std::milli>(std::chrono::high_resolution_clock::now().time_since_epoch()).count(); }
#define DISP(call) do{ double _t=now(); (call).wait(); T_disp+=now()-_t; }while(0)

static inline double nw3(){return 0;}
// reused scratch: RZ=resize(no zero, fully overwritten), ZB=zero whole (padded)
#define RZv(v,n) static vector<float> v; v.resize(n);
#define RZb(v,n) static vector<bf16> v; v.resize(n);
int GLOBAL[4]={7,15,23,31};
bool isglob(int li){ for(int g:GLOBAL) if(li==g) return true; return false; }

// global handles
H hln,hqkv_w,hqkv_g,ho_w,ho_g,hqt_w,hqt_g,hsm_w,hsm_g,hpv_w,hpv_g,hf1,hgelu,hf2,hf1v2,hflashw;
// LN bos
xrt::bo lin,lout;
// proj scratch
xrt::bo qkv_wA,qkv_wC,qkv_gA,qkv_gC,o_wA,o_wC,o_gA,o_gC;
// attn bos (win/glob): qA,qB,SC(shared sm-in),P(shared pv-in),pB,pC
xrt::bo qaW,qbW,scW,pW,pbW,pcW; long NBW;
xrt::bo qaF,qbF,pbF,OF;
xrt::bo qaG,qbG,scG,pG,pbG,pcG; long NBG;
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

// attention: q,k,v [G,S,64] (q,k post-rope). win: Sp=S=576 nomask. glob: Sp=1344 S=1296 mask.
void attn(vector<float>&q,vector<float>&k,vector<float>&v,bool glob,vector<float>&O){
  int G=glob?16:64, S=glob?1296:576, Sp=glob?1344:576; long NB=glob?NBG:NBW;
  if(!glob){ // window flash attention: fused qkt+softmax+pv, 1 dispatch, Q unscaled (flash folds 1/sqrt(d))
    static vector<bf16> Qf,Kf,Vf,Ob_; Qf.assign(G*S*d,0);Kf.assign(G*S*d,0);Vf.assign(G*S*d,0);
    { int total=G*S*d;
      _Pragma("omp parallel for schedule(static)") for(int i=0;i<total;i+=16){
        int n=std::min(16,total-i);
        __m512 vq=_mm512_loadu_ps(q.data()+i),vk=_mm512_loadu_ps(k.data()+i),vv=_mm512_loadu_ps(v.data()+i);
        _mm256_storeu_si256((__m256i*)(Qf.data()+i),(__m256i)_mm512_cvtneps_pbh(vq));
        _mm256_storeu_si256((__m256i*)(Kf.data()+i),(__m256i)_mm512_cvtneps_pbh(vk));
        _mm256_storeu_si256((__m256i*)(Vf.data()+i),(__m256i)_mm512_cvtneps_pbh(vv)); (void)n; } }
    qaF.write(Qf.data()); qaF.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    qbF.write(Kf.data()); qbF.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    pbF.write(Vf.data()); pbF.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    DISP(hflashw.k(3,hflashw.bi,hflashw.nw,qaF,qbF,pbF,OF));
    Ob_.assign(G*S*d,0); OF.sync(XCL_BO_SYNC_BO_FROM_DEVICE); OF.read(Ob_.data());
    O.resize(G*S*d);
    for(size_t i=0;i<(size_t)G*S*d;i++) O[i]=b2f(Ob_[i]);
    return;
  }
  H&hqt=glob?hqt_g:hqt_w; H&hsm=glob?hsm_g:hsm_w; H&hpv=glob?hpv_g:hpv_w;
  xrt::bo&qa=glob?qaG:qaW,&qb=glob?qbG:qbW,&sc=glob?scG:scW,&P=glob?pG:pW,&pb=glob?pbG:pbW,&pc=glob?pcG:pcW;
  float scale=1.f/std::sqrt((float)d);
  // pack q*scale,k,v into [G,Sp,d] bf16 (pad rows for glob)
  static vector<bf16> Q,K,V; Q.assign(G*Sp*d,0);K.assign(G*Sp*d,0);V.assign(G*Sp*d,0);
  #pragma omp parallel for collapse(2) schedule(static)
  for(int g=0;g<G;g++)for(int s=0;s<S;s++)for(int c=0;c<d;c++){
    Q[(g*Sp+s)*d+c]=f2b(q[(g*S+s)*d+c]*scale);
    K[(g*Sp+s)*d+c]=f2b(k[(g*S+s)*d+c]);
    V[(g*Sp+s)*d+c]=f2b(v[(g*S+s)*d+c]); }
  qa.write(Q.data()); qa.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  qb.write(K.data()); qb.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  pb.write(V.data()); pb.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  DISP(hqt.k(3,hqt.bi,hqt.nw,qa,qb,sc));  // qkt -> sc (bf16)
  if(glob){ // mask cols>=S to -1e4
    vector<bf16> s(NB); sc.sync(XCL_BO_SYNC_BO_FROM_DEVICE); sc.read(s.data());
    for(long g=0;g<G;g++)for(int r=0;r<Sp;r++)for(int cc=S;cc<Sp;cc++) s[(g*Sp+r)*Sp+cc]=f2b(-1e4f);
    sc.write(s.data()); sc.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  }
  DISP(hsm.k(3,hsm.bi,hsm.nw,sc,P));  // softmax sc->P
  if(glob){ vector<bf16> p(NB); P.sync(XCL_BO_SYNC_BO_FROM_DEVICE); P.read(p.data());
    for(long g=0;g<G;g++){ for(int r=0;r<Sp;r++)for(int cc=S;cc<Sp;cc++) p[(g*Sp+r)*Sp+cc]=0; for(int r=S;r<Sp;r++)for(int cc=0;cc<Sp;cc++) p[(g*Sp+r)*Sp+cc]=0; }
    P.write(p.data()); P.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  }
  DISP(hpv.k(3,hpv.bi,hpv.nw,P,pb,pc));  // pv -> pc (f32 [G,Sp,d])
  RZv(Of,G*Sp*d); pc.sync(XCL_BO_SYNC_BO_FROM_DEVICE); pc.read(Of.data());
  O.resize(G*S*d);
  for(int g=0;g<G;g++)for(int s=0;s<S;s++)for(int c=0;c<d;c++) O[(g*S+s)*d+c]=Of[(g*Sp+s)*d+c];
}

// one block: x [1296,C] -> [1296,C]
void block(vector<float>&x,int li){
  bool glob=isglob(li);
  static vector<float> res; res=x;
  vector<float> xn; npu_ln(x,ln1w[li],ln1b[li],xn);  // [1296,C]
  
  int Mp = glob?1296:2304;
  int Mpad = glob?1536:2304;
  // build proj input xflat [Mp,C]
  static vector<float> xflat;
  if(!glob){ // window partition 36x36 -> pad48 -> 4x[24,24]=[2304,C]
    xflat.assign(2304*C,0.f);
    for(int w0=0;w0<2;w0++)for(int w1=0;w1<2;w1++)for(int i=0;i<24;i++)for(int j=0;j<24;j++){
      int gi=w0*24+i, gj=w1*24+j; int win=w0*2+w1; int row=win*576+i*24+j;
      if(gi<36&&gj<36){ const float*s=&xn[(gi*36+gj)*C]; std::memcpy(&xflat[row*C],s,C*4); }
    }
  } else xflat=xn;
  // qkvproj
  H&hq=glob?hqkv_g:hqkv_w; xrt::bo&qA=glob?qkv_gA:qkv_wA; xrt::bo&qCo=glob?qkv_gC:qkv_wC;
  static vector<float> xin; xin.assign(Mpad*C,0.f); std::memcpy(xin.data(),xflat.data(),Mp*C*4);
  wbf_v(qA,xin);
  DISP(hq.k(3,hq.bi,hq.nw,qA,WB_qkv[li],qCo));
  RZv(qkv,Mpad*3072); qCo.sync(XCL_BO_SYNC_BO_FROM_DEVICE); qCo.read(qkv.data());
  // split + bias + head reshape
  int G=glob?16:64, S=glob?1296:576;
  static vector<float> q,k,v; q.resize(G*S*d);k.resize(G*S*d);v.resize(G*S*d);
  const vector<float>&bq=bqkv[li];
  if(!glob){
    // qkv[Mp=2304,3072]; token t in window win=t/576, pos p=t%576; head h; q[win*16+h, p, c]
    #pragma omp parallel for schedule(static)
    for(int t=0;t<2304;t++){ int win=t/576,p=t%576; const float*qrow=&qkv[t*3072];
      for(int h=0;h<nH;h++)for(int c=0;c<d;c++){
        int g=win*nH+h; float qq=qrow[h*d+c]+bq[h*d+c]; float kk=qrow[C+h*d+c]+bq[C+h*d+c]; float vv=qrow[2*C+h*d+c]+bq[2*C+h*d+c];
        q[(g*S+p)*d+c]=qq; k[(g*S+p)*d+c]=kk; v[(g*S+p)*d+c]=vv; } }
  } else {
    #pragma omp parallel for schedule(static)
    for(int t=0;t<1296;t++){ const float*qrow=&qkv[t*3072];
      for(int h=0;h<nH;h++)for(int c=0;c<d;c++){
        int g=h; float qq=qrow[h*d+c]+bq[h*d+c]; float kk=qrow[C+h*d+c]+bq[C+h*d+c]; float vv=qrow[2*C+h*d+c]+bq[2*C+h*d+c];
        q[(g*1296+t)*d+c]=qq; k[(g*1296+t)*d+c]=kk; v[(g*1296+t)*d+c]=vv; } }
  }
  rope(q,G,S,glob?ropeGc:ropeWc,glob?ropeGs:ropeWs);
  rope(k,G,S,glob?ropeGc:ropeWc,glob?ropeGs:ropeWs);
  vector<float> O; attn(q,k,v,glob,O);  // [G,S,d]
  // O -> standard [Mp,C]
  RZv(Ostd,Mp*C);
  if(!glob){ _Pragma("omp parallel for schedule(static)") for(int t=0;t<2304;t++){ int win=t/576,p=t%576; for(int h=0;h<nH;h++)for(int c=0;c<d;c++) Ostd[t*C+h*d+c]=O[((win*nH+h)*576+p)*d+c]; } }
  else { for(int t=0;t<1296;t++)for(int h=0;h<nH;h++)for(int c=0;c<d;c++) Ostd[t*C+h*d+c]=O[((h)*1296+t)*d+c]; }
  // oproj
  H&ho=glob?ho_g:ho_w; xrt::bo&oA=glob?o_gA:o_wA; xrt::bo&oCo=glob?o_gC:o_wC;
  static vector<float> oin; oin.assign(Mpad*C,0.f); std::memcpy(oin.data(),Ostd.data(),Mp*C*4);
  wbf_v(oA,oin);
  DISP(ho.k(3,ho.bi,ho.nw,oA,WB_o[li],oCo));
  RZv(ao,Mpad*C); oCo.sync(XCL_BO_SYNC_BO_FROM_DEVICE); oCo.read(ao.data());
  // +Ob
  const vector<float>&ob=Ob[li];
  // window unpartition -> [1296,C]
  static vector<float> attn_out; attn_out.assign(1296*C,0.f);
  if(!glob){
    for(int win=0;win<4;win++){ int w0=win/2,w1=win%2; for(int i=0;i<24;i++)for(int j=0;j<24;j++){ int gi=w0*24+i,gj=w1*24+j; if(gi<36&&gj<36){ int t=win*576+i*24+j; for(int c=0;c<C;c++) attn_out[(gi*36+gj)*C+c]=ao[t*C+c]+ob[c]; } } }
  } else { for(int t=0;t<1296;t++)for(int c=0;c<C;c++) attn_out[t*C+c]=ao[t*C+c]+ob[c]; }
  
  // residual1
  #pragma omp parallel for schedule(static)
  for(int i=0;i<1296*C;i++) x[i]=res[i]+attn_out[i];
  
  // ffn
  static vector<float> res2; res2=x;
  vector<float> xn2; npu_ln(x,ln2w[li],ln2b[li],xn2);
  static vector<float> ffa; ffa.assign(MFFN*C,0.f); std::memcpy(ffa.data(),xn2.data(),1296*C*4);
  wbf_v(f1A,ffa);
  wbf_v(f1Av2,ffa);
  DISP(hf1v2.k(3,hf1v2.bi,hf1v2.nw,f1Av2,WB_w1full[li],f1Cfull));
  f1Cfull.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  const float*hid_=f1Cfull.map<float*>(); // zero-copy: direct BO pointer
  // +b1 + gelu -> gin bf16
  { // Tier3: bias+gelu+f2b fused on CPU — skips NPU gelu dispatch (~85ms → ~19ms)
    static vector<bf16> gb; if(gb.size()<(size_t)MFFN*Hpad)gb.resize((size_t)MFFN*Hpad); const float*br=b1[li].data();
    _Pragma("omp parallel for schedule(static)") for(int r=0;r<MFFN;r++){ const float*hr=hid_+r*Hpad; bf16*ob=&gb[r*Hpad];
      for(int cc=0;cc<Hpad;cc++) ob[cc]=f2b(gelu1(hr[cc]+br[cc])); }
    gsh.write(gb.data(),(size_t)MFFN*Hpad*2,0); gsh.sync(XCL_BO_SYNC_BO_TO_DEVICE); }
  // hgelu dispatch skipped (CPU gelu faster)
  DISP(hf2.k(3,hf2.bi,hf2.nw,gsh,WB_w2[li],f2C));
  f2C.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  const float*fo_=reinterpret_cast<const float*>(f2C.map<float*>());
  const vector<float>&f2b_=fc2b[li];
  _Pragma("omp parallel for schedule(static)") for(int r=0;r<1296;r++){ const float*rr=&res2[r*C]; const float*fr=fo_+r*C; float*xr=&x[r*C]; int cc=0;
    for(;cc+16<=C;cc+=16){ __m512 v=_mm512_add_ps(_mm512_add_ps(_mm512_loadu_ps(rr+cc),_mm512_loadu_ps(fr+cc)),_mm512_loadu_ps(f2b_.data()+cc)); _mm512_storeu_ps(xr+cc,v);}
    for(;cc<C;cc++) xr[cc]=rr[cc]+fr[cc]+f2b_[cc]; }
}

static string input_file="";
static string output_file="";

int main(int argc,char**argv){
  int n_runs=3;
  for(int i=1;i<argc;i++){
    string a=argv[i];
    if(a=="--input"&&i+1<argc)  { input_file=argv[++i]; }
    else if(a=="--output"&&i+1<argc){ output_file=argv[++i]; }
    else if(a=="--runs"&&i+1<argc)  { n_runs=atoi(argv[++i]); }
  }
  DEV=xrt::device(0);
  const string A="/home/amd/project/npu_iron/sam3_attn/";
  hln=loadx(A+"layernorm/S1296");
  hqkv_w=loadx(A+"proj_mc_v2/qkvproj_w"); hqkv_g=loadx(A+"proj_mc_v2/qkvproj_g");
  ho_w=loadx(A+"proj_mc_v2/oproj_w"); ho_g=loadx(A+"proj_mc_v2/oproj_g");
  hqt_w=loadx(A+"attn_mc/qkt_bmm_w_bf16"); hqt_g=loadx(A+"attn_mc/qkt_bmm_g_bf16"); hflashw=loadx(A+"attn_v2/flash_w");
  hsm_w=loadx(A+"attn_mc/sm_batch_S576"); hsm_g=loadx(A+"attn_mc/sm_batch_S1344");
  hpv_w=loadx(A+"attn_mc/pv_bmm_w"); hpv_g=loadx(A+"attn_mc/pv_bmm_g");
  hf1=loadx(A+"ffn_mc/ffn1_half"); hgelu=loadx(A+"gelu_mc"); hf2=loadx(A+"ffn_mc_v2/ffn2"); hf1v2=loadx(A+"ffn_mc_v2/ffn1");
  fprintf(stderr,"xclbins loaded\n");
  lin=mkbo(hln,3,(size_t)Sp_ln*C*2); lout=mkbo(hln,4,(size_t)Sp_ln*C*2);
  qkv_wA=mkbo(hqkv_w,3,(size_t)2304*C*2); qkv_wC=mkbo(hqkv_w,5,(size_t)2304*3072*4);
  qkv_gA=mkbo(hqkv_g,3,(size_t)1536*C*2); qkv_gC=mkbo(hqkv_g,5,(size_t)1536*3072*4);
  o_wA=mkbo(ho_w,3,(size_t)2304*C*2); o_wC=mkbo(ho_w,5,(size_t)2304*C*4);
  o_gA=mkbo(ho_g,3,(size_t)1536*C*2); o_gC=mkbo(ho_g,5,(size_t)1536*C*4);
  NBW=(long)64*576*576; NBG=(long)16*1344*1344;
  qaW=mkbo(hqt_w,3,(size_t)64*576*d*2); qbW=mkbo(hqt_w,4,(size_t)64*576*d*2); scW=mkbo(hqt_w,5,(size_t)NBW*2);
  qaF=mkbo(hflashw,3,(size_t)64*576*d*2); qbF=mkbo(hflashw,4,(size_t)64*576*d*2); pbF=mkbo(hflashw,5,(size_t)64*576*d*2); OF=mkbo(hflashw,6,(size_t)64*576*d*2);
  pW=mkbo(hsm_w,4,(size_t)NBW*2); pbW=mkbo(hpv_w,4,(size_t)64*576*d*2); pcW=mkbo(hpv_w,5,(size_t)64*576*d*4);
  qaG=mkbo(hqt_g,3,(size_t)16*1344*d*2); qbG=mkbo(hqt_g,4,(size_t)16*1344*d*2); scG=mkbo(hqt_g,5,(size_t)NBG*2);
  pG=mkbo(hsm_g,4,(size_t)NBG*2); pbG=mkbo(hpv_g,4,(size_t)16*1344*d*2); pcG=mkbo(hpv_g,5,(size_t)16*1344*d*4);
  f1A=mkbo(hf1,3,(size_t)MFFN*C*2); f1C=mkbo(hf1,5,(size_t)MFFN*Nhalf*4); f1Av2=mkbo(hf1v2,3,(size_t)MFFN*C*2); f1Cfull=mkbo(hf1v2,5,(size_t)MFFN*Hpad*4);
  gin=mkbo(hgelu,3,(size_t)MFFN*Hpad*2); gsh=mkbo(hgelu,4,(size_t)MFFN*Hpad*2); f2C=mkbo(hf2,5,(size_t)MFFN*C*4);
  fprintf(stderr,"bos allocated\n");
  // load weights resident
  for(int li=0;li<32;li++){ bool glob=isglob(li); char b[64];
    auto Wq=loadf(CBB+"L"+std::to_string(li)+"_Wqkv"); WB_qkv[li]=mkbo(glob?hqkv_g:hqkv_w,4,(size_t)C*3072*2); wbf_v(WB_qkv[li],Wq);
    auto Wo=loadf(CBB+"L"+std::to_string(li)+"_Ow"); WB_o[li]=mkbo(glob?ho_g:ho_w,4,(size_t)C*C*2); wbf_v(WB_o[li],Wo);
    auto W1=loadf(CBB+"L"+std::to_string(li)+"_W1"); // [C,Hpad]
    WB_w1a[li]=mkbo(hf1,4,(size_t)C*Nhalf*2); WB_w1b[li]=mkbo(hf1,4,(size_t)C*Nhalf*2); WB_w1full[li]=mkbo(hf1v2,4,(size_t)C*Hpad*2); wbf_v(WB_w1full[li],W1);
    { vector<float> h0(C*Nhalf),h1(C*Nhalf); for(int r=0;r<C;r++){ std::memcpy(&h0[r*Nhalf],&W1[r*Hpad],Nhalf*4); std::memcpy(&h1[r*Nhalf],&W1[r*Hpad+Nhalf],Nhalf*4);} wbf_v(WB_w1a[li],h0); wbf_v(WB_w1b[li],h1); }
    auto W2=loadf(CBB+"L"+std::to_string(li)+"_W2"); WB_w2[li]=mkbo(hf2,4,(size_t)Hpad*C*2); wbf_v(WB_w2[li],W2);
    bqkv[li]=loadf(CBB+"L"+std::to_string(li)+"_bqkv"); Ob[li]=loadf(CBB+"L"+std::to_string(li)+"_Ob");
    ln1w[li]=loadf(CBB+"L"+std::to_string(li)+"_ln1w"); ln1b[li]=loadf(CBB+"L"+std::to_string(li)+"_ln1b");
    ln2w[li]=loadf(CBB+"L"+std::to_string(li)+"_ln2w"); ln2b[li]=loadf(CBB+"L"+std::to_string(li)+"_ln2b");
    b1[li]=loadf(CBB+"L"+std::to_string(li)+"_b1"); fc2b[li]=loadf(CBB+"L"+std::to_string(li)+"_fc2b");
  }
  ropeWc=loadf(CBB+"rope_win_cos"); ropeWs=loadf(CBB+"rope_win_sin");
  ropeGc=loadf(CBB+"rope_glob_cos"); ropeGs=loadf(CBB+"rope_glob_sin");
  fprintf(stderr,"weights resident\n");
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
      T_disp=0; double t0=now();
      for(int li=0;li<32;li++) block(x,li);
      double wall=now()-t0;
      fprintf(stderr,"wall=%.0fms dispatch=%.0fms\n",wall,T_disp); fflush(stderr);

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
  for(int r=0;r<n_runs;r++){ T_disp=0; double t0=now(); x=x0; for(int li=0;li<32;li++) block(x,li); double wall=now()-t0;
    printf("run%d: wall=%.0fms (%.2f FPS)  dispatch=%.0fms\n",r,wall,1000.0/wall,T_disp);
    if(r==0&&!output_file.empty()){
      FILE*fp=fopen(output_file.c_str(),"wb");
      if(fp){ fwrite(x.data(),4,x.size(),fp); fclose(fp); }
    }
  }
  return 0;
}

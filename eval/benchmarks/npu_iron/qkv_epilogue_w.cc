// qkv_epilogue_w.cc: Window QKV proj epilogue: bias+RoPE+f2b
// Input float32 tile [ROWS=32, COLS=128], bias[COLS], cos/sin[ROWS,HEAD_DIM=64]
// Output bf16 [ROWS, COLS]
#include <aie_api/aie.hpp>
#include <stdint.h>
#ifndef EPI_ROWS
#define EPI_ROWS 32
#endif
#ifndef EPI_COLS
#define EPI_COLS 128  // 2 heads × HEAD_DIM
#endif
#ifndef HEAD_DIM
#define HEAD_DIM 64
#endif

extern "C" {
// Q or K: bias + RoPE + f2b
void epilogue_qk(float*     c_tile,   // [ROWS, COLS] float32
                 float*     bias,     // [COLS] float32
                 bfloat16*  cos_lut,  // [ROWS, HEAD_DIM] bf16
                 bfloat16*  sin_lut,  // [ROWS, HEAD_DIM] bf16
                 bfloat16*  out) {    // [ROWS, COLS] bf16
  event0();
  for (int r = 0; r < EPI_ROWS; r++) {
    float*    ci  = c_tile  + r * EPI_COLS;
    bfloat16* oi  = out     + r * EPI_COLS;
    const bfloat16* co = cos_lut + r * HEAD_DIM;
    const bfloat16* si = sin_lut + r * HEAD_DIM;
    for (int h = 0; h < EPI_COLS / HEAD_DIM; h++) {
      float*    ch = ci + h * HEAD_DIM;
      bfloat16* oh = oi + h * HEAD_DIM;
      const float*    bh = bias + h * HEAD_DIM;
      for (int i = 0; i < HEAD_DIM; i += 2) {
        float q0 = ch[i]   + bh[i];
        float q1 = ch[i+1] + bh[i+1];
        float c0 = (float)co[i],   c1 = (float)co[i+1];
        float s0 = (float)si[i],   s1 = (float)si[i+1];
        oh[i]   = (bfloat16)(q0 * c0 - q1 * s0);
        oh[i+1] = (bfloat16)(q1 * c1 + q0 * s1);
      }
    }
  }
  event1();
}

// V: bias + f2b only (no RoPE)
void epilogue_v(float*    c_tile,
                float*    bias,
                bfloat16* out) {
  event0();
  for (int r = 0; r < EPI_ROWS; r++) {
    float*    ci = c_tile + r * EPI_COLS;
    bfloat16* oi = out    + r * EPI_COLS;
    for (int c = 0; c < EPI_COLS; c++) {
      oi[c] = (bfloat16)(ci[c] + bias[c]);
    }
  }
  event1();
}
}

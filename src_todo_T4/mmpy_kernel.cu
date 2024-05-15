// ;-*- mode: c;-*-
// Matrix multiply device code
#include "../src/types.h"
#include "../src/utils.h"
#include "mytypes.h"
#include <assert.h>
#include <math.h>
using namespace std;

#include <stdio.h>

#ifdef NAIVE
__global__ void matMul(int N, _FTYPE_ *C, _FTYPE_ *A, _FTYPE_ *B) {

  int I = blockIdx.y * blockDim.y + threadIdx.y;
  int J = blockIdx.x * blockDim.x + threadIdx.x;

  if ((I < N) && (J < N)) {
    _FTYPE_ _c = 0;
    for (unsigned int k = 0; k < N; k++) {
      _FTYPE_ a = A[I * N + k];
      _FTYPE_ b = B[k * N + J];
      _c += a * b;
    }
    C[I * N + J] = _c;
  }
}

#else
// You should be changing the kernel here for the non naive implementation.
__global__ void matMul(int N, _FTYPE_ *C, _FTYPE_ *A, _FTYPE_ *B) {
  extern __shared__ _FTYPE_ As_Bs[];

  _FTYPE_ *As = (_FTYPE_ *)As_Bs;
  _FTYPE_ *Bs = (_FTYPE_ *)As_Bs + TILEDIM_M * TILEDIM_K;

  const int ty = threadIdx.y, tx = threadIdx.x;
  const int by = blockIdx.y, bx = blockIdx.x;

  // int startI = by * TILEDIM_K;
  // int startJ = bx * TILEDIM_K;

  _FTYPE_ Cij[TILESCALE_N * TILESCALE_M] = {0.0f};
  // #pragma unroll
  // for (int cij = 0; cij < TILESCALE_M * TILESCALE_N; ++cij)
  // Cij[cij] = 0;
  const int tid = tx + ty * blockDim.x;

  for (int kk = 0; kk < (N / TILEDIM_K + (N % TILEDIM_K != 0)); kk++) {

    if ((tid / TILEDIM_K < N) && (tid % TILEDIM_K + kk * TILEDIM_K) < N)
      As[(tid / TILEDIM_K) * TILEDIM_K + tid % TILEDIM_K] =
          A[(by * TILEDIM_N + tid / TILEDIM_K) * N + tid % TILEDIM_K +
            kk * TILEDIM_K];
    else
      As[(tid / TILEDIM_K) * TILEDIM_K + tid % TILEDIM_K] = 0;

    if ((tid / TILEDIM_N + kk * TILEDIM_K) < N && (tid % TILEDIM_N) < N)
      Bs[(tid / TILEDIM_N) * TILEDIM_N + tid % TILEDIM_N] =
          B[(tid / TILEDIM_N + kk * TILEDIM_K) * N + tid % TILEDIM_N +
            bx * TILEDIM_M];
    else
      Bs[(tid / TILEDIM_N) * TILEDIM_N + tid % TILEDIM_N] = 0;

    // Thread Block
    // #pragma unroll
    //     for (int ci = 0; ci < TILESCALE_M; ++ci) {
    // #pragma unroll
    //       for (int cj = 0; cj < TILESCALE_N; ++cj) {
    //         int tyn = ty + cj * TILEDIM_N / TILESCALE_N;
    //         int txn = tx + ci * TILEDIM_M / TILESCALE_M;
    //         int I = startI + tyn;
    //         int J = startJ + txn;

    //         if (I < N && kk * TILEDIM_M + txn < N)
    //           As[tyn * TILEDIM_M + txn] = A[I * N + kk * TILEDIM_M + txn];
    //         else
    //           As[tyn * TILEDIM_M + txn] = 0;

    //         if (kk * TILEDIM_M + tyn < N && J < N)
    //           Bs[tyn * TILEDIM_N + txn] = B[(kk * TILEDIM_M + tyn) * N + J];
    //         else
    //           Bs[tyn * TILEDIM_N + txn] = 0;
    //       }
    //     }
    __syncthreads();
    // Warp Tile

    // Thread Tile
    // #pragma unroll
    for (int k = 0; k < TILEDIM_K; k++) {
      // #pragma unroll
      for (int ci = 0; ci < TILESCALE_M; ++ci) {
        // #pragma unroll
        for (int cj = 0; cj < TILESCALE_N; ++cj) {
          //           int tyn = ty + cj * TILEDIM_N / TILESCALE_N;
          //           int txn = tx + ci * TILEDIM_M / TILESCALE_M;
          Cij[cj * TILESCALE_M + ci] +=
              As[(ty * TILESCALE_N + cj) * TILEDIM_K + k] *
              Bs[(k)*TILEDIM_N + tx * TILESCALE_M + ci];
          // if (tx == 0 & ty == 0)
          //   printf("A %f idx=%d,B %f idx=%d \n",
          //          A[(ty * TILESCALE_N + cj) * TILEDIM_K + k],
          //          ((ty * TILESCALE_N + cj) * TILEDIM_K + k),
          //          B[(k)*TILEDIM_N + tx * TILESCALE_M + ci],
          //          ((k)*TILEDIM_N + tx * TILESCALE_M + ci));
        }
      }
    }
    __syncthreads();
    // if ((tid / TILEDIM_N + kk * TILEDIM_K) < N && (tid % TILEDIM_N) < N)
    //   printf("%d kk=%d, %f\n", tid,kk,
    //          Bs[(tid / TILEDIM_N) * TILEDIM_N + tid % TILEDIM_N]);
    // if ((tid / TILEDIM_K < N) && (tid % TILEDIM_K + kk * TILEDIM_K) < N)
    //   printf("%d kk=%d, %f\n", tid, kk,
    //          As[(tid / TILEDIM_K) * TILEDIM_K + tid % TILEDIM_K]);
  }

#pragma unroll
  for (int ci = 0; ci < TILESCALE_M; ++ci) {
#pragma unroll
    for (int cj = 0; cj < TILESCALE_N; ++cj) {
      // int tyn = ty + cj * TILEDIM_N / TILESCALE_N;
      // int txn = tx + ci * TILEDIM_M / TILESCALE_M;
      // int I = startI + tyn;
      // int J = startJ + txn;
      // if (I < N && J < N)
      if ((TILESCALE_M * ty + cj + by * TILEDIM_N) < N &&
          (ci + tx * TILESCALE_M + bx * TILEDIM_M) < N)
        C[(TILESCALE_M * ty + cj + by * TILEDIM_N) * N + ci + tx * TILESCALE_M +
          bx * TILEDIM_M] = Cij[cj * TILESCALE_M + ci];
    }
  }
}
#endif

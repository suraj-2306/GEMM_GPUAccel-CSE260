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
  extern __shared__ double As_Bs[];

  double *As = (double *)As_Bs;
  double *Bs = (double *)As_Bs + TILEDIM_M * TILEDIM_K;

  int ty = threadIdx.y, tx = threadIdx.x;
  int by = blockIdx.y, bx = blockIdx.x;

  int startI = by * TILEDIM_K;
  int startJ = bx * TILEDIM_K;

  double Cij[TILESCALE_N][TILESCALE_M];
#pragma unroll
  for (int ci = 0; ci < TILESCALE_M; ++ci)
#pragma unroll
    for (int cj = 0; cj < TILESCALE_N; ++cj)
      Cij[cj][ci] = 0;

  for (int kk = 0; kk < (N / TILEDIM_M + (N % TILEDIM_M != 0)); kk++) {

    // Thread Block
#pragma unroll
    for (int ci = 0; ci < TILESCALE_M; ++ci) {
#pragma unroll
      for (int cj = 0; cj < TILESCALE_N; ++cj) {
        int tyn = ty + cj * TILEDIM_N / TILESCALE_N;
        int txn = tx + ci * TILEDIM_M / TILESCALE_M;
        int I = startI + tyn;
        int J = startJ + txn;

        if (I < N && kk * TILEDIM_M + txn < N)
          As[tyn * TILEDIM_M + txn] = A[I * N + kk * TILEDIM_M + txn];
        else
          As[tyn * TILEDIM_M + txn] = 0;

        if (kk * TILEDIM_M + tyn < N && J < N)
          Bs[tyn * TILEDIM_N + txn] = B[(kk * TILEDIM_M + tyn) * N + J];
        else
          Bs[tyn * TILEDIM_N + txn] = 0;
      }
    }

    __syncthreads();
    // Warp Tile

    // Thread Tile
#pragma unroll
    for (int k = 0; k < TILEDIM_K; k++) {
#pragma unroll
      for (int ci = 0; ci < TILESCALE_M; ++ci) {
#pragma unroll
        for (int cj = 0; cj < TILESCALE_N; ++cj) {
          int tyn = ty + cj * TILEDIM_N / TILESCALE_N;
          int txn = tx + ci * TILEDIM_M / TILESCALE_M;
          Cij[cj][ci] += As[tyn * TILEDIM_M + k] * Bs[k * TILEDIM_N + txn];
        }
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int ci = 0; ci < TILESCALE_M; ++ci) {
#pragma unroll
    for (int cj = 0; cj < TILESCALE_N; ++cj) {
      int tyn = ty + cj * TILEDIM_N / TILESCALE_N;
      int txn = tx + ci * TILEDIM_M / TILESCALE_M;
      int I = startI + tyn;
      int J = startJ + txn;
      if (I < N && J < N)
        C[I * N + J] = Cij[cj][ci];
    }
  }
}
#endif

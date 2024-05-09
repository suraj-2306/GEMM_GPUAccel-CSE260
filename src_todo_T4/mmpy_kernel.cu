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
  __shared__ double As[TILEDIM_M][TILEDIM_K], Bs[TILEDIM_K][TILEDIM_N];

  int ty = threadIdx.y, tx = threadIdx.x;
  int by = blockIdx.y, bx = blockIdx.x;

  int startI = by * TILEDIM_K;
  int J = bx * TILEDIM_K + tx;

  double Cij[TILESCALE_N];
  for (int cj = 0; cj < TILESCALE_N; ++cj)
    Cij[cj] = 0;

  for (int kk = 0; kk < (N / TILEDIM_M + (N % TILEDIM_M != 0)); kk++) {

#pragma unroll
    for (int cj = 0; cj < TILESCALE_N; ++cj) {
      int tyn = ty + cj * TILEDIM_N / TILESCALE_N;
      int I = startI + tyn;

      if ((I) < N && kk * TILEDIM_M + tx < N)
        As[tyn][tx] = A[(I)*N + kk * TILEDIM_M + tx];
      else
        As[tyn][tx] = 0;

      if (kk * TILEDIM_M + tyn < N && J < N)
        Bs[tyn][tx] = B[(kk * TILEDIM_M + tyn) * N + J];
      else
        Bs[tyn][tx] = 0;
    }

    __syncthreads();
#pragma unroll
    for (int k = 0; k < TILEDIM_K; k++) {
#pragma unroll
      for (int cj = 0; cj < TILESCALE_N; ++cj) {
        int tyn = ty + cj * TILEDIM_N / TILESCALE_N;
        Cij[cj] += As[tyn][k] * Bs[k][tx];
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int cj = 0; cj < TILESCALE_N; ++cj) {
    int I = startI + ty + cj * TILEDIM_N / TILESCALE_N;
    if (I < N && J < N)
      C[I * N + J] = Cij[cj];
  }
}
#endif

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

  int I = by * TILEDIM_K + ty;
  int J = bx * TILEDIM_K + tx;

  double Cij[2] = {0, 0};
  for (int kk = 0; kk < (N / TILEDIM_M + (N % TILEDIM_M != 0)); kk++) {
    if (I < N && kk * TILEDIM_M + tx < N)
      As[ty][tx] = A[I * N + kk * TILEDIM_M + tx];
    else
      As[ty][tx] = 0;

    if ((I + TILEDIM_N / TILESCALE_N) < N && kk * TILEDIM_M + tx < N)
      As[ty + TILEDIM_N / TILESCALE_N][tx] =
          A[(I + TILEDIM_N / TILESCALE_N) * N + kk * TILEDIM_M + tx];
    else
      As[ty + TILEDIM_N / TILESCALE_N][tx] = 0;

    if (kk * TILEDIM_M + ty < N && J < N)
      Bs[ty][tx] = B[(kk * TILEDIM_M + ty) * N + J];
    else
      Bs[ty][tx] = 0;

    if (kk * TILEDIM_M + ty + TILEDIM_N / TILESCALE_N < N && J < N)
      Bs[ty + TILEDIM_N / TILESCALE_N][tx] =
          B[(kk * TILEDIM_M + ty + TILEDIM_N / TILESCALE_N) * N + J];
    else
      Bs[ty + TILEDIM_N / TILESCALE_N][tx] = 0;

    __syncthreads();
#pragma unroll
    for (int k = 0; k < TILEDIM_K; k++) {
      Cij[0] += As[ty][k] * Bs[k][tx];
      Cij[1] += As[ty + TILEDIM_N / TILESCALE_N][k] * Bs[k][tx];
    }
    __syncthreads();
  }

  if (I < N && J < N)
    C[I * N + J] = Cij[0];
  if (I + TILEDIM_N / TILESCALE_N < N && J < N)
    C[(I + TILEDIM_N / TILESCALE_N) * N + J] = Cij[1];
}
#endif

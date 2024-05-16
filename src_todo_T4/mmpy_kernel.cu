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

  const int tid = tx + ty * blockDim.x;

#pragma unroll
  for (int kk = 0; kk < (N / TILEDIM_K + (N % TILEDIM_K != 0)); kk++) {

    for (int tij = 0; tij < TLOAD; tij++) {
      if (((tid * TLOAD + tij) / TILEDIM_K + by * TILEDIM_N) < N &&
          ((tid * TLOAD + tij) % TILEDIM_K + kk * TILEDIM_K) < N)
        As[((tid * TLOAD + tij) / TILEDIM_K) * TILEDIM_K +
           (tid * TLOAD + tij) % TILEDIM_K] =
            A[(by * TILEDIM_N + (tid * TLOAD + tij) / TILEDIM_K) * N +
              (tid * TLOAD + tij) % TILEDIM_K + kk * TILEDIM_K];
      else
        As[((tid * TLOAD + tij) / TILEDIM_K) * TILEDIM_K +
           (tid * TLOAD + tij) % TILEDIM_K] = 0;

      if (((tid * TLOAD + tij) / TILEDIM_N + kk * TILEDIM_K) < N &&
          ((tid * TLOAD + tij) % TILEDIM_N + bx * TILEDIM_M) < N)
        Bs[((tid * TLOAD + tij) / TILEDIM_N) * TILEDIM_N +
           (tid * TLOAD + tij) % TILEDIM_N] =
            B[((tid * TLOAD + tij) / TILEDIM_N + kk * TILEDIM_K) * N +
              (tid * TLOAD + tij) % TILEDIM_N + bx * TILEDIM_M];
      else
        Bs[((tid * TLOAD + tij) / TILEDIM_N) * TILEDIM_N +
           (tid * TLOAD + tij) % TILEDIM_N] = 0;
    }

    __syncthreads();
    // Warp Tile

    // Thread Tile
#pragma unroll
    for (int k = 0; k < TILEDIM_K; k++) {
#pragma unroll
      for (int ci = 0; ci < TILESCALE_M; ++ci) {
        _FTYPE_ bscopy = Bs[(k)*TILEDIM_N + tx * TILESCALE_M + ci];
#pragma unroll
        for (int cj = 0; cj < TILESCALE_N; ++cj) {
          Cij[cj * TILESCALE_M + ci] +=
              As[(ty * TILESCALE_N + cj) * TILEDIM_K + k] * bscopy;
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
      if ((TILESCALE_M * ty + cj + by * TILEDIM_N) < N &&
          (ci + tx * TILESCALE_M + bx * TILEDIM_M) < N)
        C[(TILESCALE_M * ty + cj + by * TILEDIM_N) * N + ci + tx * TILESCALE_M +
          bx * TILEDIM_M] = Cij[cj * TILESCALE_M + ci];
    }
  }
}
#endif

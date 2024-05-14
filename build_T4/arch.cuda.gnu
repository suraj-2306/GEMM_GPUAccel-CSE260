ifeq ($(USER),raj)
  GENCODE_SM75  : -arch=sm_50 \ 
-gencode=arch=compute_50,code=sm_50 \ 
-gencode=arch=compute_52,code=sm_52 \ 
-gencode=arch=compute_60,code=sm_60 \ 
-gencode=arch=compute_61,code=sm_61 \ 
-gencode=arch=compute_70,code=sm_70 \ 
-gencode=arch=compute_75,code=sm_75 \
-gencode=arch=compute_75,code=compute_75

  CUDA_INSTALL_PATH=/opt/cuda
  LIB_BLAS  = -lcblas -lpthread -lm
endif

ifeq ($(USER),ubuntu)
  GENCODE_SM75  :=-gencode arch=compute_75,code=sm_75
  CUDA_INSTALL_PATH=/usr/local/cuda-11.6
  LIB_BLAS  = -lblas -lpthread -lm
  C++FLAGS += -DAWS
endif


GENCODE_FLAGS :=$(GENCODE_SM75)
PTXFLAGS=-v
# PTXFLAGS=-dlcm=ca 
NVCCFLAGS= -O3 $(GENCODE_FLAGS) -c  -lineinfo

# Compilers
NVCC            = $(shell which nvcc)
C++             = $(shell which g++)
C++LINK         = $(C++)
NVCCLINK        = $(NVCC)
CLINK           = $(CC)

.SUFFIXES:
.SUFFIXES: .cpp .c .cu .o

.cpp.o:
		$(C++) $(C++FLAGS) -c $<

.c.o:
		$(C++) $(C++FLAGS) -c $<

.cu.o:
	$(NVCC)  $(NVCCFLAGS) $<


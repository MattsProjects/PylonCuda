# Makefile for Basler pylon sample program
.PHONY: all clean

# The program to build
NAME       := PylonCuda
CUDANAME   := cuda_kernels

# Installation directories for pylon
PYLON_ROOT ?= /opt/pylon

# CUDA directories
CUDA_ROOT_DIR=/usr/local/cuda-10.2
CUDA_INC_DIR= -I$(CUDA_ROOT_DIR)/targets/aarch64-linux/include
CUDA_LIB_DIR= -L$(CUDA_ROOT_DIR)/targets/aarch64-linux/lib
CUDA_INC_COM= -I$(CUDA_ROOT_DIR)/samples/common/inc
CUDA_LIB_COM= -L$(CUDA_ROOT_DIR)/samples/common/lib

# NVCC compiler options:
NVCC=nvcc
NVCC_FLAGS=
NVCC_LIBS=

# CUDA linking libraries
CUDA_LINK_LIBS= -lcudart

# Build tools and flags
LD         := $(CXX)
CPPFLAGS   := $(shell $(PYLON_ROOT)/bin/pylon-config --cflags) $(CUDA_INC_DIR) $(CUDA_INC_COM) -std=c++11 -DLINUX_BUILD
CXXFLAGS   := #e.g., CXXFLAGS=-g -O0 for debugging
LDFLAGS    := $(shell $(PYLON_ROOT)/bin/pylon-config --libs-rpath) $(CUDA_LIB_DIR) $(CUDA_LIB_COM)
LDLIBS     := $(shell $(PYLON_ROOT)/bin/pylon-config --libs) $(CUDA_LINK_LIBS)

# Rules for building: make output directory, make program, move to output directory
all: $(NAME)
 
# Link c++ and CUDA compiled object files to target executable:
$(NAME) : $(CUDANAME).o $(NAME).o
	$(CXX) $(LDFLAGS) $(CC_FLAGS) $(OBJS) -o $@ $^ $(LDLIBS) $(CUDA_INC_DIR) $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS)

# compile C++ source files to object files:	
$(NAME).o: $(NAME).cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c -o $@ $<

# Compile CUDA source files to object files:
$(CUDANAME).o : $(CUDANAME).cu $(CUDANAME).h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ $(NVCC_LIBS)

clean:
	$(RM) *.o $(NAME)

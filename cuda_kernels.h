// Declarations of wrapper functions that help
// bring cuda code into existing C++ application

#include <stdint.h>
#include <cmath>

int CudaInitialize(int device = 0);

void CudaAllocateBuffer(void** pBuffer, int bufSize);

void CudaFreeBuffer(void* pBuffer);

void CudaBlurMono8(uint8_t* input, uint8_t* output, int width, int height);

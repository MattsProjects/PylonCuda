// Grab_UsingBufferFactory.cpp
/*
	Note: Before getting started, Basler recommends reading the "Programmer's Guide" topic
	in the pylon C++ API documentation delivered with pylon.
	If you are upgrading to a higher major version of pylon, Basler also
	strongly recommends reading the "Migrating from Previous Versions" topic in the pylon C++ API documentation.

	This sample demonstrates how to use a user-provided buffer factory.
	Using a buffer factory is optional and intended for advanced use cases only.
	A buffer factory is only necessary if you want to grab into externally supplied buffers.
*/

// Includ files for CUDA
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Include files to use the pylon API.
#include <pylon/PylonIncludes.h>
#ifdef PYLON_WIN_BUILD
#include <pylon/PylonGUI.h>
#endif

// Namespace for using pylon objects.
using namespace Pylon;

// Namespace for using cout.
using namespace std;

// Number of images to be grabbed.
static const uint32_t c_countOfImagesToGrab = 100;

// Maximum number of buffers to allocate for grabbing
static const uint32_t c_maxNumBuffers = 5;

// A user-provided buffer factory.
class MyBufferFactory : public IBufferFactory
{
public:
	MyBufferFactory() : m_lastBufferContext(1000)
	{
	}

	virtual ~MyBufferFactory()
	{
	}

	// Will be called when the Instant Camera object needs to allocate a buffer.
	// Return the buffer and context data in the output parameters.
	// In case of an error, new() will throw an exception
	// which will be forwarded to the caller to indicate an error.
	// Warning: This method can be called by different threads.
	virtual void AllocateBuffer(size_t bufferSize, void** pCreatedBuffer, intptr_t& bufferContext)
	{
		try
		{
			// Allocate buffer for pixel data.
			// If you already have a buffer allocated by your image processing library, you can use this instead.
			// In this case, you must modify the delete code (see below) accordingly

			// Use Cuda unified memory: https://developer.nvidia.com/blog/unified-memory-cuda-beginners/
			cudaMallocManaged((void**)pCreatedBuffer, sizeof(uint8_t) * bufferSize);

			bufferContext = ++m_lastBufferContext;
			cout << "Created buffer " << bufferContext << ", " << pCreatedBuffer << ", Size: " << bufferSize << endl;
		}
		catch (const std::exception&)
		{
			// In case of an error you must free the memory you may have already allocated.
			if (*pCreatedBuffer != NULL)
			{
				cudaFree(pCreatedBuffer);

				*pCreatedBuffer = NULL;
			}

			// Rethrow exception.
			// AllocateBuffer can also just return with *pCreatedBuffer = NULL to indicate
			// that no buffer is available at the moment.
			throw;
		}
	}

	// Frees a previously allocated buffer.
	// Warning: This method can be called by different threads.
	virtual void FreeBuffer(void* pCreatedBuffer, intptr_t bufferContext)
	{
		cudaFree(pCreatedBuffer);

		cout << "Freed buffer " << bufferContext << ", " << pCreatedBuffer << endl;
	}

	// Destroys the buffer factory.
	// This will be used when you pass the ownership of the buffer factory instance to pylon
	// by defining Cleanup_Delete. pylon will call this function to destroy the instance
	// of the buffer factory. If you don't pass the ownership to pylon (Cleanup_None),
	// this method will be ignored.
	virtual void DestroyBufferFactory()
	{
		delete this;
	}

protected:

	unsigned long m_lastBufferContext;
};

// example functions to rotate image
__device__ uint8_t readPixVal(uint8_t* ImgSrc, int ImgWidth, int x, int y)
{
	return (uint8_t)ImgSrc[y * ImgWidth + x];
}
__device__ void putPixVal(uint8_t* ImgSrc, int ImgWidth, int x, int y, float floatVal)
{
	ImgSrc[y * ImgWidth + x] = floatVal;
}
__global__ void Rotate(uint8_t* Source, uint8_t* Destination, int sizeX, int sizeY, float deg)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;// Kernel definition
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int xc = sizeX - sizeX / 2;
	int yc = sizeY - sizeY / 2;
	int newx = ((float)i - xc) * cos(deg) - ((float)j - yc) * sin(deg) + xc;
	int newy = ((float)i - xc) * sin(deg) + ((float)j - yc) * cos(deg) + yc;
	if (newx >= 0 && newx < sizeX && newy >= 0 && newy < sizeY)
	{
		putPixVal(Destination, sizeX, i, j, readPixVal(Source, sizeX, newx, newy));
	}
}

int main(int /*argc*/, char* /*argv*/[])
{
	// The exit code of the sample application.
	int exitCode = 0;

	// Before using any pylon methods, the pylon runtime must be initialized.
	PylonInitialize();

	try
	{
		// Choose which GPU to run on, change this on a multi-GPU system.
		cudaError_t cudaStatus = cudaSetDevice(0);
		if (cudaStatus != cudaSuccess)
		{
			cout << "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?" << endl;
			return 1;
		}

		// The buffer factory must be created first because objects on the
		// stack are destroyed in reverse order of creation.
		// The buffer factory must exist longer than the Instant Camera object
		// in this sample.
		std::cout << "Creating Buffer Factory" << std::endl;
		MyBufferFactory myFactory;

		// Create an instant camera object with the camera device found first.
		CInstantCamera camera(CTlFactory::GetInstance().CreateFirstDevice());

		// Print the model name of the camera.
		cout << "Using device " << camera.GetDeviceInfo().GetModelName() << endl;

		camera.Open();

		GenApi::CIntegerPtr(camera.GetNodeMap().GetNode("Width"))->SetValue(960);
		GenApi::CIntegerPtr(camera.GetNodeMap().GetNode("Height"))->SetValue(960);

		// Use our own implementation of a buffer factory.
		// Since we control the lifetime of the factory object, we pass Cleanup_None.
		camera.SetBufferFactory(&myFactory, Cleanup_None);

		// The parameter MaxNumBuffer can be used to control the count of buffers
		// allocated for grabbing. The default value of this parameter is 10.
		camera.MaxNumBuffer = c_maxNumBuffers;

		// Start the grabbing of c_countOfImagesToGrab images.
		// The camera device is parameterized with a default configuration which
		// sets up free-running continuous acquisition.
		camera.StartGrabbing(c_countOfImagesToGrab);

		// Camera.StopGrabbing() is called automatically by the RetrieveResult() method
		// when c_countOfImagesToGrab images have been retrieved.
		while (camera.IsGrabbing())
		{
			// This smart pointer will receive the grab result data.
			CGrabResultPtr ptrGrabResult;

			// Wait for an image and then retrieve it. A timeout of 5000 ms is used.
			camera.RetrieveResult(5000, ptrGrabResult, TimeoutHandling_ThrowException);

			// Image grabbed successfully?
			if (ptrGrabResult == NULL)
				cout << "Grab Result Pointer is NULL!" << endl;
			else
			{
				if (ptrGrabResult->GrabSucceeded())
				{
					cout << endl;
					// Access the image data.
					cout << "Context: " << ptrGrabResult->GetBufferContext() << endl;
					cout << "SizeX: " << ptrGrabResult->GetWidth() << endl;
					cout << "SizeY: " << ptrGrabResult->GetHeight() << endl;
					const uint8_t* pImageBuffer = (uint8_t*)ptrGrabResult->GetBuffer();
					cout << "First value of pixel data: " << (uint32_t)pImageBuffer[0] << endl;

					uint8_t* pSrcBuf = (uint8_t*)ptrGrabResult->GetBuffer();
					uint8_t* pOutBuf = NULL;
					cudaMallocManaged((void**)pOutBuf, sizeof(uint8_t) * ptrGrabResult->GetPayloadSize());

					Rotate<<<1, 1>>>(pSrcBuf, pOutBuf, ptrGrabResult->GetWidth(), ptrGrabResult->GetHeight(), 90);

					CPylonImage rotatedImage;
					rotatedImage.AttachUserBuffer(pOutBuf, ptrGrabResult->GetPayloadSize(), ptrGrabResult->GetPixelType(), ptrGrabResult->GetHeight(), ptrGrabResult->GetWidth(), ptrGrabResult->GetPaddingX());


#ifdef PYLON_WIN_BUILD
					// Display the grabbed image.
					Pylon::DisplayImage(1, ptrGrabResult);
					Pylon::DisplayImage(2, rotatedImage);
#endif
					cudaFree(pOutBuf);
				}
				else
				{
					cout << "Error: " << std::hex << ptrGrabResult->GetErrorCode() << std::dec << " " << ptrGrabResult->GetErrorDescription();
				}
			}
		}
	}
	catch (const GenericException& e)
	{
		// Error handling.
		cerr << "An exception occurred." << endl
			<< e.GetDescription() << endl;
		exitCode = 1;
	}

	// Comment the following two lines to disable waiting on exit.
	cerr << endl << "Press enter to exit." << endl;
	while (cin.get() != '\n');

	// Releases all pylon resources.
	PylonTerminate();

	return exitCode;
}

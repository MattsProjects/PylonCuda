/*
	Note: Before getting started, Basler recommends reading the "Programmer's Guide" topic
	in the pylon C++ API documentation delivered with pylon.
	If you are upgrading to a higher major version of pylon, Basler also
	strongly recommends reading the "Migrating from Previous Versions" topic in the pylon C++ API documentation.

	This sample demonstrates how to use a user-provided buffer factory.
	Using a buffer factory is optional and intended for advanced use cases only.
	A buffer factory is only necessary if you want to grab into externally supplied buffers.
*/

#include "cuda_kernels.h"

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
static const uint32_t c_countOfImagesToGrab = 10;

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
			// You can use "Pinned Memory" on the host.
			// https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/
			// Or if your system supports "Unified Memory" on the GPU, cudaMallocManaged() can be used instead
			// https://developer.nvidia.com/blog/unified-memory-cuda-beginners/
			// This wrapper function will check and pick the best one available
			CudaAllocateBuffer(pCreatedBuffer, bufferSize);

			bufferContext = ++m_lastBufferContext;
			cout << "Created buffer " << bufferContext << ", " << pCreatedBuffer << ", Size: " << bufferSize << endl;
		}
		catch (const std::exception&)
		{
			// In case of an error you must free the memory you may have already allocated.
			if (*pCreatedBuffer != NULL)
			{
				// Free memory wrapper function
				CudaFreeBuffer(pCreatedBuffer);

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
		// Free memory wrapper function
		CudaFreeBuffer(pCreatedBuffer);

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



int main(int /*argc*/, char* /*argv*/[])
{
	// The exit code of the sample application.
	int exitCode = 0;

	// Before using any pylon methods, the pylon runtime must be initialized.
	PylonInitialize();

	// Choose the first (best) GPU available.
	if (CudaInitialize(0) == 1)
		return 1;

	try
	{
		// The buffer factory must be created first because objects on the
		// stack are destroyed in reverse order of creation.
		// The buffer factory must exist longer than the Instant Camera object
		// in this sample.
		MyBufferFactory myFactory;

		// Create an instant camera object with the camera device found first.
		CInstantCamera camera(CTlFactory::GetInstance().CreateFirstDevice());

		// Print the model name of the camera.
		cout << "Using device " << camera.GetDeviceInfo().GetModelName() << endl;

		camera.Open();

		// setup the camera
		GenApi::CEnumerationPtr(camera.GetNodeMap().GetNode("PixelFormat"))->FromString("Mono8");
		GenApi::CEnumerationPtr(camera.GetNodeMap().GetNode("TriggerSelector"))->FromString("FrameStart");
		GenApi::CEnumerationPtr(camera.GetNodeMap().GetNode("TriggerSource"))->FromString("Software");
		GenApi::CEnumerationPtr(camera.GetNodeMap().GetNode("TriggerMode"))->FromString("On");
		GenApi::CIntegerPtr(camera.GetNodeMap().GetNode("Width"))->SetValue(GenApi::CIntegerPtr(camera.GetNodeMap().GetNode("Width"))->GetMax());
		GenApi::CIntegerPtr(camera.GetNodeMap().GetNode("Height"))->SetValue(GenApi::CIntegerPtr(camera.GetNodeMap().GetNode("Height"))->GetMax());

		// Use our own implementation of a buffer factory.
		// Since we control the lifetime of the factory object, we pass Cleanup_None.
		camera.SetBufferFactory(&myFactory, Cleanup_None);

		// The parameter MaxNumBuffer can be used to control the count of buffers
		// allocated for grabbing. The default value of this parameter is 10.
		camera.MaxNumBuffer = c_maxNumBuffers;

		// Start the grabbing of c_countOfImagesToGrab images.
		// The camera device is parameterized with a default configuration which
		// sets up free-running continuous acquisition.
		camera.StartGrabbing();

		int imagesGrabbed = 0;

		while (imagesGrabbed < c_countOfImagesToGrab)
		{
			// This smart pointer will receive the grab result data.
			CGrabResultPtr ptrGrabResult;

			// trigger the camera
			camera.ExecuteSoftwareTrigger();

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
					int width = ptrGrabResult->GetWidth();
					int height = ptrGrabResult->GetHeight();
					cout << "Image   : " << imagesGrabbed << endl;
					cout << "Context : " << ptrGrabResult->GetBufferContext() << endl;
					cout << "SizeX   : " << width << endl;
					cout << "SizeY   : " << height << endl;

					cout << "Saving Grabbed Image..." << endl;
					std::string fileName = "image_";
					fileName.append(std::to_string(imagesGrabbed));
					fileName.append(".png");
					CImagePersistence::Save(Pylon::ImageFileFormat_Png, fileName.c_str(), ptrGrabResult);

					cout << "Blurring image..." << endl;
					uint8_t* pImage = (uint8_t*)ptrGrabResult->GetBuffer();
					CPylonImage blurredImage;
					uint8_t* pBlurred = new uint8_t[width * height];

					// Use the GPU To blur the image :-)
					CudaBlurMono8(pImage, pBlurred, width, height);

					blurredImage.AttachUserBuffer(pBlurred, width * height, ptrGrabResult->GetPixelType(), width, height, ptrGrabResult->GetPaddingX());

					cout << "Saving Blurred Image..." << endl;
					fileName = "image_";
					fileName.append(std::to_string(imagesGrabbed));
					fileName.append("_blurred");
					fileName.append(".png");
					CImagePersistence::Save(Pylon::ImageFileFormat_Png, fileName.c_str(), blurredImage);

#ifdef PYLON_WIN_BUILD
					// Display the grabbed image.
					Pylon::DisplayImage(1, ptrGrabResult);
					Pylon::DisplayImage(2, blurredImage);
#endif
					delete[] pBlurred;

					imagesGrabbed++;
				}
				else
				{
					cout << "Error: " << std::hex << ptrGrabResult->GetErrorCode() << std::dec << " " << ptrGrabResult->GetErrorDescription();
				}
			}
		}
		camera.StopGrabbing();
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

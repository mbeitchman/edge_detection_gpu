#include <stdio.h>
#include <Windows.h>
#include <math.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

RGBTRIPLE *g_nimageCPU  = NULL;
RGBTRIPLE *g_nimageGPUG = NULL;
RGBTRIPLE *g_nimageGPUS = NULL;
LONG g_imagesize = 0;

__global__ void GlobalMemoryLumaKernel(RGBTRIPLE* dev_origimage, RGBTRIPLE* dev_grayimage, LONG width)
{
	unsigned int x = (blockIdx.x *blockDim.x) + threadIdx.x;
	unsigned int y = (blockIdx.y *blockDim.y) + threadIdx.y;
	unsigned int i = (y*width+x);

	BYTE gray = (dev_origimage[i].rgbtBlue * 0.11f) + (dev_origimage[i].rgbtRed * 0.3f) + (dev_origimage[i].rgbtGreen * 0.59f);
	dev_grayimage[i].rgbtBlue = dev_grayimage[i].rgbtRed = dev_grayimage[i].rgbtGreen = gray;
}

__global__ void GlobalMemorySobelKernel(RGBTRIPLE* dev_grayimage, RGBTRIPLE* dev_procimage, LONG width)
{
	unsigned int x = (blockIdx.x *blockDim.x) + threadIdx.x;
	unsigned int y = (blockIdx.y *blockDim.y) + threadIdx.y;
	unsigned int i = (y*width+x);

	// sobel
	BYTE a, b, c, d ,e, f, g, h;
	a = b = c = d = e = f = g = h = 0;

	a = dev_grayimage[((y-1) *width) + (x-1)].rgbtGreen;
	b = dev_grayimage[((y-1) *width) + x].rgbtGreen;
	c = dev_grayimage[((y-1) *width) + (x+1)].rgbtGreen;
	d = dev_grayimage[(y *width) + (x-1)].rgbtGreen;
	e = dev_grayimage[(y *width) + (x+1)].rgbtGreen;
	f = dev_grayimage[((y+1) *width) + (x-1)].rgbtGreen;
	g = dev_grayimage[((y+1) *width) + x].rgbtGreen;
	h = dev_grayimage[((y+1) *width) + (x+1)].rgbtGreen;

	float gx = (c-a) + (2*e) - (2*d) + (h-f);
	float gy = (f-a) + (2*g) - (2*b) + (h-c);

	float result = sqrt((gx*gx)+(gy*gy));
	if(result > 255)
	{
		result = 255;
	}
		
	dev_procimage[i].rgbtBlue = dev_procimage[i].rgbtRed = dev_procimage[i].rgbtGreen = (BYTE)(255-result);
}

__global__ void SharedMemorySobelKernel(RGBTRIPLE* dev_grayimage, RGBTRIPLE* dev_procimage, LONG width)
{
	unsigned int x = (blockIdx.x *blockDim.x) + threadIdx.x;
	unsigned int xleft = ((blockIdx.x-1) *blockDim.x) + threadIdx.x;
	unsigned int xright = ((blockIdx.x+1) *blockDim.x) + threadIdx.x;
	unsigned int yabove = ((blockIdx.y-1) *blockDim.y) + threadIdx.y;
	unsigned int ybelow = ((blockIdx.y+1) *blockDim.y) + threadIdx.y;
	unsigned int y = (blockIdx.y *blockDim.y) + threadIdx.y;
	unsigned int i = (y*width+x);

	unsigned int tx = threadIdx.x;
	unsigned int ty = threadIdx.y;

	__shared__ BYTE LocalBlock[48][48];

	// read data into shared memory
	LocalBlock[ty][tx] = dev_grayimage[yabove*width+xleft].rgbtBlue;
	LocalBlock[ty][tx+16] = dev_grayimage[yabove*width+x].rgbtBlue;
	LocalBlock[ty][tx+32] = dev_grayimage[yabove*width+xright].rgbtBlue;

	LocalBlock[ty+16][tx] = dev_grayimage[y*width+xleft].rgbtBlue;
	LocalBlock[ty+16][tx+16] = dev_grayimage[y*width+x].rgbtBlue;
	LocalBlock[ty+16][tx+32] = dev_grayimage[y*width+xright].rgbtBlue;

	LocalBlock[ty+32][tx] = dev_grayimage[ybelow*width+xleft].rgbtBlue;
	LocalBlock[ty+32][tx+16] = dev_grayimage[ybelow*width+x].rgbtBlue;
	LocalBlock[ty+32][tx+32] = dev_grayimage[ybelow*width+xright].rgbtBlue;

	__syncthreads();

	// sobel
	float result = 0;
	BYTE a, b, c, d ,e, f, g, h;
	a = b = c = d = e = f = g = h = 0;

	tx+=16;
	ty+=16;

	a = LocalBlock[ty-1][tx-1];
	b = LocalBlock[ty-1][tx];
	c = LocalBlock[ty-1][tx+1];
	d = LocalBlock[ty][tx-1];
	e = LocalBlock[ty][tx+1];
	f = LocalBlock[ty+1][tx-1];
	g = LocalBlock[ty+1][tx];
	h = LocalBlock[ty+1][tx+1];

	float gx = (c-a) + (2*e) - (2*d) + (h-f);
	float gy = (f-a) + (2*g) - (2*b) + (h-c);

	result = sqrt((gx*gx)+(gy*gy));
	if(result > 255)
	{
		result = 255;
	}

	// copy back to global memory
	dev_procimage[i].rgbtBlue = dev_procimage[i].rgbtRed = dev_procimage[i].rgbtGreen = (BYTE)(255-result);
}

__global__ void SharedMemorySobelKernelAlt(RGBTRIPLE* dev_grayimage, RGBTRIPLE* dev_procimage, LONG width)
{
	unsigned int x = (blockIdx.x *blockDim.x) + threadIdx.x;
	unsigned int xleft = ((blockIdx.x-1) *blockDim.x)+15;
	unsigned int xright = ((blockIdx.x+1)) *blockDim.x;
	
	unsigned int y = (blockIdx.y *blockDim.y) + threadIdx.y;
	unsigned int yabove = ((blockIdx.y-1) *blockDim.y) + 15;
	unsigned int ybelow = ((blockIdx.y+1) *blockDim.y);

	unsigned int i = (y*width+x);

	unsigned int tx = threadIdx.x;
	unsigned int ty = threadIdx.y;
	
	// read data into shared memory
	__shared__ BYTE LocalBlock[18][18];
	LocalBlock[ty+1][tx+1] = dev_grayimage[i].rgbtBlue;

	if(tx == 0)
	{
		LocalBlock[ty][tx] = dev_grayimage[y*width+xleft].rgbtBlue;
	}
	else if(tx == 15)
	{
		LocalBlock[ty][tx+2] = dev_grayimage[y*width+xright].rgbtBlue;
	}

	if(ty == 0)
	{
		LocalBlock[ty][tx] = dev_grayimage[yabove*width+x].rgbtBlue;
	}
	
	if(ty == 15)
	{
		LocalBlock[ty+2][tx] = dev_grayimage[ybelow*width+x].rgbtBlue;
	}

	unsigned int xleft1 = ((blockIdx.x-1) *blockDim.x);
	unsigned int xright1 = ((blockIdx.x+1) *blockDim.x);
	unsigned int yabove1 ((blockIdx.y-1) *blockDim.y);
	unsigned int ybelow1 ((blockIdx.y+1) *blockDim.y);
	LocalBlock[0][0] = dev_grayimage[(yabove1*width)+xleft1+255].rgbtBlue;
	LocalBlock[0][17] = dev_grayimage[yabove*width+xright1+239].rgbtBlue;
	LocalBlock[17][0] = dev_grayimage[ybelow1*width+xleft1+15].rgbtBlue;
	LocalBlock[17][17] = dev_grayimage[ybelow1*width+xright1].rgbtBlue;

	__syncthreads();

	// sobel
	float result = 0;
	BYTE a, b, c, d ,e, f, g, h;
	a = b = c = d = e = f = g = h = 0;

	tx+=1;
	ty+=1;

	a = LocalBlock[ty-1][tx-1];
	b = LocalBlock[ty-1][tx];
	c = LocalBlock[ty-1][tx+1];
	d = LocalBlock[ty][tx-1];
	e = LocalBlock[ty][tx+1];
	f = LocalBlock[ty+1][tx-1];
	g = LocalBlock[ty+1][tx];
	h = LocalBlock[ty+1][tx+1];

	float gx = (c-a) + (2*e) - (2*d) + (h-f);
	float gy = (f-a) + (2*g) - (2*b) + (h-c);

	result = sqrt((gx*gx)+(gy*gy));
	if(result > 255)
	{
		result = 255;
	}


	// copy back to global memory
	dev_procimage[i].rgbtBlue = dev_procimage[i].rgbtRed = dev_procimage[i].rgbtGreen = (BYTE)(255-result);
}

void DoEdgeDetectionOnGPUWithGlobalMemory(char* input, char* output, BOOL bTimeCalculation)
{
	DWORD written;
    BITMAPFILEHEADER bfh;
    BITMAPINFOHEADER bih;
	RGBTRIPLE* dev_grayimage = 0;
	RGBTRIPLE* dev_procimage = 0;
	RGBTRIPLE* dev_origimage = 0;
	LARGE_INTEGER startTime;
	LARGE_INTEGER endTime;
	cudaError_t cudaStatus;

	HANDLE hfile = CreateFile(input, GENERIC_READ,FILE_SHARE_READ,NULL,OPEN_EXISTING,NULL,NULL);

    // Read the header
    ReadFile(hfile,&bfh,sizeof(bfh),&written,NULL);
    ReadFile(hfile,&bih,sizeof(bih),&written,NULL);

    // Read image
    g_imagesize = bih.biWidth*bih.biHeight; 
    RGBTRIPLE *image = new RGBTRIPLE[g_imagesize]; 
    ReadFile(hfile,image,g_imagesize*sizeof(RGBTRIPLE),&written,NULL); // Reads it off the disk
    CloseHandle(hfile);

	size_t size = g_imagesize*sizeof(RGBTRIPLE);

	// start time
	if(bTimeCalculation)
	{
		QueryPerformanceCounter(&startTime);
	}

	do
	{
		cudaStatus = cudaSetDevice(0);
		if(cudaStatus != cudaSuccess)
		{
			printf("DoEdgeDetectionOnGPUWithGlobalMemory() failed in cudaSetDevice()\n");
		}
	
		// allocate memory on device for source image and copy source image to the device
		cudaStatus = cudaMalloc((void**)&dev_origimage, size);
		if(cudaStatus != cudaSuccess)
		{
			printf("DoEdgeDetectionOnGPUWithGlobalMemory() failed in cudaMalloc()\n");
		}

		cudaStatus = cudaMemcpy((void*)dev_origimage, image, size, cudaMemcpyHostToDevice);
		if(cudaStatus != cudaSuccess)
		{
			printf("DoEdgeDetectionOnGPUWithGlobalMemory() failed in cudaMemcpy()\n");
		}

		// allocate memory in device for gray image
		cudaStatus = cudaMalloc((void**)&dev_grayimage, size);
		if(cudaStatus != cudaSuccess)
		{
			printf("DoEdgeDetectionOnGPUWithGlobalMemory() failed in cudaMalloc()\n");
		}

		// allocate memory in device for final image
		cudaStatus = cudaMalloc((void**)&dev_procimage, size);
		if(cudaStatus != cudaSuccess)
		{
			printf("DoEdgeDetectionOnGPUWithGlobalMemory() failed in cudaMalloc()\n");
		}

		dim3 dimBlock(16, 16);
		dim3 dimGrid(bih.biWidth / dimBlock.x, bih.biHeight / dimBlock.y);

		GlobalMemoryLumaKernel<<<dimGrid, dimBlock>>>(dev_origimage, dev_grayimage, bih.biWidth);
		GlobalMemorySobelKernel<<<dimGrid, dimBlock>>>(dev_grayimage, dev_procimage, bih.biWidth);

		cudaStatus = cudaDeviceSynchronize();
		if(cudaStatus != cudaSuccess)
		{
			printf("DoEdgeDetectionOnGPUWithGlobalMemory() failed in cudaDeviceSynchronize()\n");
		}

		// copy processed image back to cpu
		g_nimageGPUG = new RGBTRIPLE[g_imagesize];
		cudaStatus = cudaMemcpy((void*)g_nimageGPUG, (void*)dev_procimage, size, cudaMemcpyDeviceToHost);
		if(cudaStatus != cudaSuccess)
		{
			printf("DoEdgeDetectionOnGPUWithGlobalMemory() failed in cudaMemcpy()\n");
		}
		
	}while(FALSE);

	cudaFree((void*)dev_procimage);
	cudaFree((void*)dev_grayimage);
	cudaFree((void*)dev_origimage);

	if(cudaStatus == cudaSuccess && bTimeCalculation)
	{
		// end time
		QueryPerformanceCounter(&endTime);

		// Copy image
		hfile = CreateFile(output,GENERIC_WRITE,FILE_SHARE_WRITE,NULL,CREATE_ALWAYS,NULL,NULL);
		WriteFile(hfile, &bfh, sizeof(bfh), &written, NULL);
		WriteFile(hfile, &bih, sizeof(bih), &written, NULL);
		WriteFile(hfile, g_nimageGPUG,g_imagesize*sizeof(RGBTRIPLE),&written,NULL); 
		CloseHandle(hfile);

		// report time
		LARGE_INTEGER freq;
		QueryPerformanceFrequency(&freq);
		if(freq.QuadPart == 0)
		{
			printf("ERROR: frequency of performance counter is 0");
		}
		else
		{
			LONGLONG timeDiff = endTime.QuadPart - startTime.QuadPart;
			float Duration = ((float) timeDiff * 1000) / (float) freq.QuadPart;
			wprintf(L"\n Total time to do edge detection on the gpu using global memory is %.5f milliseconds.\n", Duration);
		}
	}

	delete image;
}

void DoEdgeDetectionOnGPUWithSharedMemory(char* input, char* output, BOOL bTimeCalculation)
{
	DWORD written;
    BITMAPFILEHEADER bfh;
    BITMAPINFOHEADER bih;
	RGBTRIPLE* dev_grayimage = 0;
	RGBTRIPLE* dev_procimage = 0;
	RGBTRIPLE* dev_origimage = 0;
	LARGE_INTEGER startTime;
	LARGE_INTEGER endTime;
	cudaError_t cudaStatus;

	HANDLE hfile = CreateFile(input, GENERIC_READ,FILE_SHARE_READ,NULL,OPEN_EXISTING,NULL,NULL);

    // Read the header
    ReadFile(hfile,&bfh,sizeof(bfh),&written,NULL);
    ReadFile(hfile,&bih,sizeof(bih),&written,NULL);

    // Read image
    g_imagesize = bih.biWidth*bih.biHeight; 
    RGBTRIPLE *image = new RGBTRIPLE[g_imagesize]; 
    ReadFile(hfile,image,g_imagesize*sizeof(RGBTRIPLE),&written,NULL); // Reads it off the disk
    CloseHandle(hfile);

	size_t size = g_imagesize*sizeof(RGBTRIPLE);

	// start time
	if(bTimeCalculation)
	{
		QueryPerformanceCounter(&startTime);
	}

	do
	{
		cudaStatus = cudaSetDevice(0);
		if(cudaStatus != cudaSuccess)
		{
			printf("DoEdgeDetectionOnGPUWithSharedMemory() failed in cudaSetDevice()\n");
		}
	
		// allocate memory on device for source image and copy source image to the device
		cudaStatus = cudaMalloc((void**)&dev_origimage, size);
		if(cudaStatus != cudaSuccess)
		{
			printf("DoEdgeDetectionOnGPUWithSharedMemory() failed in cudaMalloc()\n");
		}

		cudaStatus = cudaMemcpy((void*)dev_origimage, image, size, cudaMemcpyHostToDevice);
		if(cudaStatus != cudaSuccess)
		{
			printf("DoEdgeDetectionOnGPUWithSharedMemory() failed in cudaMemcpy()\n");
		}

		// allocate memory in device for gray image
		cudaStatus = cudaMalloc((void**)&dev_grayimage, size);
		if(cudaStatus != cudaSuccess)
		{
			printf("DoEdgeDetectionOnGPUWithSharedMemory() failed in cudaMalloc()\n");
		}

		// allocate memory in device for final image
		cudaStatus = cudaMalloc((void**)&dev_procimage, size);
		if(cudaStatus != cudaSuccess)
		{
			printf("DoEdgeDetectionOnGPUWithSharedMemory() failed in cudaMalloc()\n");
		}

		dim3 dimBlock(16, 16);
		dim3 dimGrid(bih.biWidth / dimBlock.x, bih.biHeight / dimBlock.y);

		GlobalMemoryLumaKernel<<<dimGrid, dimBlock>>>(dev_origimage, dev_grayimage, bih.biWidth);

		SharedMemorySobelKernel<<<dimGrid, dimBlock>>>(dev_grayimage, dev_procimage, bih.biWidth);

		cudaStatus = cudaDeviceSynchronize();
		if(cudaStatus != cudaSuccess)
		{
			printf("DoEdgeDetectionOnGPUWithSharedMemory() failed in cudaDeviceSynchronize()\n");
		}

		// copy processed image back to cpu
		g_nimageGPUS = new RGBTRIPLE[g_imagesize];
		cudaStatus = cudaMemcpy((void*)g_nimageGPUS, (void*)dev_procimage, size, cudaMemcpyDeviceToHost);
		if(cudaStatus != cudaSuccess)
		{
			printf("DoEdgeDetectionOnGPUWithSharedMemory() failed in cudaMemcpy()\n");
		}

	}while(FALSE);

	cudaFree((void*)dev_procimage);
	cudaFree((void*)dev_grayimage);
	cudaFree((void*)dev_origimage);
	
	if(cudaStatus == cudaSuccess && bTimeCalculation)
	{
		// end time
		QueryPerformanceCounter(&endTime);

		// Copy image
		hfile = CreateFile(output,GENERIC_WRITE,FILE_SHARE_WRITE,NULL,CREATE_ALWAYS,NULL,NULL);
		WriteFile(hfile, &bfh, sizeof(bfh), &written, NULL);
		WriteFile(hfile, &bih, sizeof(bih), &written, NULL);
		WriteFile(hfile, g_nimageGPUS,g_imagesize*sizeof(RGBTRIPLE),&written,NULL); 
		CloseHandle(hfile);

		// report time
		LARGE_INTEGER freq;
		QueryPerformanceFrequency(&freq);
		if(freq.QuadPart == 0)
		{
			printf("ERROR: frequency of performance counter is 0");
		}
		else
		{
			LONGLONG timeDiff = endTime.QuadPart - startTime.QuadPart;
			float Duration = ((float) timeDiff * 1000) / (float) freq.QuadPart; 
			wprintf(L"\nTotal time to do edge detection on the gpu using shared memory is %.5f milliseconds.\n", Duration);
		}
	}

	delete image;
}

void DoEdgeDetectionOnCPU(char* input, char* output, BOOL bTimeCalculation)
{
	DWORD written;
	LARGE_INTEGER startTime;
	LARGE_INTEGER endTime;
    BITMAPFILEHEADER bfh;
    BITMAPINFOHEADER bih;

    // Open the file
    HANDLE hfile = CreateFile(input,GENERIC_READ,FILE_SHARE_READ,NULL,OPEN_EXISTING,NULL,NULL);

    // Read the header
    ReadFile(hfile,&bfh,sizeof(bfh),&written,NULL);
    ReadFile(hfile,&bih,sizeof(bih),&written,NULL);

    // Read image
    g_imagesize = bih.biWidth*bih.biHeight; 
    RGBTRIPLE *image = new RGBTRIPLE[g_imagesize]; 
    ReadFile(hfile,image,g_imagesize*sizeof(RGBTRIPLE),&written,NULL); // Reads it off the disk

    CloseHandle(hfile);

	// start time
	if(bTimeCalculation)
	{
		QueryPerformanceCounter(&startTime);
	}

    //convert to gray scale
	RGBTRIPLE* gimage = new RGBTRIPLE[g_imagesize]; 
    for(LONG h=0; h<bih.biHeight; h++)
	{
        for(LONG w=0; w<bih.biWidth; w++)
		{
            RGBTRIPLE triple = image[h*bih.biWidth + w];
			BYTE luminance = (triple.rgbtBlue * 0.11) + (triple.rgbtGreen * 0.59) + (triple.rgbtRed * 0.3);
			
			triple.rgbtBlue = triple.rgbtGreen = triple.rgbtRed = luminance;

            gimage[h*bih.biWidth + w] = triple;
        }
    }

	g_nimageCPU = new RGBTRIPLE[g_imagesize]; 
	// sobel
	for(LONG height=0; height<bih.biHeight; height++)
	{
        for(LONG w=0; w<bih.biWidth; w++)
		{	
			BYTE a, b, c, d , e, f, g, h = 0;

			if(height - 1 >= 0 && w -1 >= 0)
			{
				a = gimage[((height-1)*bih.biWidth) + w-1].rgbtBlue;
			}

			if(height-1 >= 0)
			{
				b = gimage[((height-1)*bih.biWidth) + w].rgbtBlue;
			}
			
			if(height-1 >= 0 && w+1 < bih.biWidth)
			{
				c = gimage[((height-1)*bih.biWidth) + (w+1)].rgbtBlue;
			}

			if(w - 1 >=0 )
			{
				d = gimage[height*bih.biWidth + (w-1)].rgbtBlue;
			}
			
			if(w + 1 < bih.biWidth)
			{
				e = gimage[height*bih.biWidth + (w-1)].rgbtBlue;
			}

			if(w - 1 >= 0 && height + 1 < bih.biHeight)
			{
				f = gimage[((height+1)*bih.biWidth) + w-1].rgbtBlue;
			}

			if(height + 1 < bih.biHeight)
			{
				g = gimage[((height+1)*bih.biWidth) + w].rgbtBlue;
			}

			if(height + 1 < bih.biHeight && w + 1 < bih.biWidth)
			{
				h = gimage[((height+1)*bih.biWidth) + (w+1)].rgbtBlue;
			}

			float gx = (c - a) + (2*e) - (2*d) + (h - f);
			float gy = (f -a) + (2*g) - (2*b) + (h - c);

			float result = sqrt((gx*gx) + (gy*gy));
			
			if(result > 255)
			{
				result = 255;
			}

			g_nimageCPU[height*bih.biWidth + w].rgbtBlue = 255 - (BYTE)result;
			g_nimageCPU[height*bih.biWidth + w].rgbtGreen = 255 - (BYTE)result;
			g_nimageCPU[height*bih.biWidth + w].rgbtRed = 255 - (BYTE)result;	
		}
	}

	QueryPerformanceCounter(&endTime);

    hfile = CreateFile(output,GENERIC_WRITE,FILE_SHARE_WRITE,NULL,CREATE_ALWAYS,NULL,NULL);
    WriteFile(hfile, &bfh, sizeof(bfh), &written, NULL);
    WriteFile(hfile, &bih, sizeof(bih), &written, NULL);
    WriteFile(hfile, g_nimageCPU,g_imagesize*sizeof(RGBTRIPLE),&written,NULL); 
    CloseHandle(hfile);

	// time
	if(bTimeCalculation)
	{
		LARGE_INTEGER freq;
		QueryPerformanceFrequency(&freq);
		if(freq.QuadPart == 0)
		{
			printf("ERROR: frequency of performance counter is 0");
		}
		else
		{
			LONGLONG timeDiff = endTime.QuadPart - startTime.QuadPart;
			float Duration = ((float) timeDiff * 1000.0) / (float) freq.QuadPart;
			  
			wprintf(L"\nTotal time to do edge detection on the cpu is %.5f milliseconds\n", Duration);
		}
	}

	delete gimage;
	delete image;
}

void CompareImages()
{
	BOOL bDiff = FALSE;
	LONG j = 0;

	if(g_nimageCPU != NULL && g_nimageGPUG != NULL &&  g_nimageGPUS == NULL)
	{
		for(LONG i = 0; i < g_imagesize; i++)
		{
			// cpu compared to gpu global
			if(g_nimageCPU[i].rgbtBlue != g_nimageGPUG[i].rgbtBlue)
			{
				j++;
				bDiff = TRUE;
			}
		}

		if(!bDiff)
		{
			printf("\n CPU and GPU-global calculations are bit exact.\n");
		}
		else
		{
			printf("\n CPU and GPU-global calculations are not bit exact (%d/%d).\n", j, g_imagesize);
		}
	}
	else if(g_nimageCPU != NULL && g_nimageGPUG != NULL &&  g_nimageGPUS != NULL)
	{
		for(LONG i = 0; i < g_imagesize; i++)
		{
			// cpu compared to gpu global
			if(g_nimageCPU[i].rgbtBlue != g_nimageGPUG[i].rgbtBlue)
			{
				j++;
				bDiff = TRUE;
			}
		}

		if(!bDiff)
		{
			printf("\n CPU and GPU-global calculations are bit exact.\n");
		}
		else
		{
			printf("\n CPU and GPU-global calculations are not bit exact (%d/%d).\n", j, g_imagesize);
		}

		j = 0;
		bDiff = FALSE;
		for(LONG i = 0; i < g_imagesize; i++)
		{
			// cpu compared to gpu shared
			if(g_nimageCPU[i].rgbtBlue != g_nimageGPUS[i].rgbtBlue)
			{
				bDiff = TRUE;
				j++;
			}
		}

		if(!bDiff)
		{
			printf("\n CPU and GPU-shared calculations are bit exact.\n");
		}
		else
		{
			printf("\n CPU and GPU-shared calculations are not bit exact (%d/%d).\n", j, g_imagesize);
		}

		j = 0;
		bDiff = FALSE;
		for(LONG i = 0; i < g_imagesize; i++) 
		{
			if(g_nimageGPUG[i].rgbtBlue != g_nimageGPUS[i].rgbtBlue)
			{
				j++;
				bDiff = TRUE;
			}
		}

		if(!bDiff)
		{
			printf("\nGPU-global and GPU-shared calculations are bit exact.\n");
		}
		else
		{
			printf("\n GPU and GPU-shared calculations are not bit exact (%d/%d).\n", j, g_imagesize);
		}
	}
}

int main(int argc, char* argv[])
{
	if(argc != 4)
	{
        printf("Usage: %s <in_bmpfile> <out_bmpfile> <computation>\n", argv[0]);
        return -1;
    }

	if(_stricmp(argv[3], "-CPU") == 0)
	{
		DoEdgeDetectionOnCPU(argv[1], argv[2], TRUE);
	}
	else if(_stricmp(argv[3], "-GPUG") == 0)
	{
		//DoEdgeDetectionOnCPU(argv[1], argv[2], FALSE);
		DoEdgeDetectionOnGPUWithGlobalMemory(argv[1], argv[2], TRUE);
		//CompareImages();
	}
	else if(_stricmp(argv[3], "-GPUS") == 0)
	{
		//DoEdgeDetectionOnCPU(argv[1], argv[2], FALSE);
		//DoEdgeDetectionOnGPUWithGlobalMemory(argv[1], argv[2], FALSE);
		DoEdgeDetectionOnGPUWithSharedMemory(argv[1], argv[2], TRUE);
		//CompareImages();
	}

	delete g_nimageCPU;
	delete g_nimageGPUS;
	delete g_nimageGPUG;

    return 0;
}

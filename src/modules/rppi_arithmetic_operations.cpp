/*
MIT License

Copyright (c) 2019 - 2024 Advanced Micro Devices, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include "rppdefs.h"
#include "rppi_validate.hpp"
#include "rppi_arithmetic_operations.h"
#include "cpu/host_arithmetic_operations.hpp"
#include <unistd.h>
#include <time.h>
#include <stdio.h>
#include <iostream>

#ifdef HIP_COMPILE
#include "rpp_hip_common.hpp"
#include "hip/hip_declarations.hpp"
#elif defined(OCL_COMPILE)
#include "rpp_cl_common.hpp"
#include "cl/cl_declarations.hpp"
#endif //backend

#define ERROR_INVALID_ENUM() do{printf("Error invalid enum. Fun: %s, para: %d\n", __FUNCTION__, para); abort(); }while(0)

/******************** add ********************/

RppStatus
rppi_add_u8_pln1_batchPD_host(RppPtr_t srcPtr1,
                              RppPtr_t srcPtr2,
                              RppiSize *srcSize,
                              RppiSize maxSrcSize,
                              RppPtr_t dstPtr,
                              Rpp32u nbatchSize,
                              rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_host_roi(roiPoints, rpp::deref(rppHandle));
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    add_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr1),
                          static_cast<Rpp8u*>(srcPtr2),
                          srcSize,
                          rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                          static_cast<Rpp8u*>(dstPtr),
                          rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                          rpp::deref(rppHandle).GetBatchSize(),
                          RPPI_CHN_PLANAR,
                          1,
                          rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

NppStatus nppiAdd_8u_C3RSfs(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                        Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor)
						
{
	int noOfImages = 1;
	int ip_channel = 3;
	RppiSize *srcSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize));
	RppiSize maxSize;
	srcSize->width  = oSizeROI.width;
	srcSize->height = oSizeROI.height;
	maxSize.width  = oSizeROI.width;
	maxSize.height = oSizeROI.height;
	
	rppHandle_t handle;
	hipStream_t stream;
	hipStreamCreate(&stream);
	rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

	RppStatus status;
	//status = rppi_add_u8_pkd3_batchPD_gpu(d_input, d_input_second, srcSize, maxSize, d_output, noOfImages, handle);
	status = rppi_add_u8_pkd3_batchPD_gpu((RppPtr_t)pSrc1, (RppPtr_t)pSrc2, srcSize, maxSize, (RppPtr_t)pDst, noOfImages, handle);

	hipDeviceSynchronize();
			
    rppDestroyGPU(handle);
	free(srcSize);

	return(hipRppStatusTocudaNppStatus(status));
}
NppStatus nppiAdd_8u_C1RSfs_Ctx(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst, 
				int nDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx)
{
    int noOfImages = 1;
    int ip_channel = 1;
    RppiSize *srcSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize));
    RppiSize maxSize;
    srcSize->width  = oSizeROI.width;
    srcSize->height = oSizeROI.height;
    maxSize.width  = oSizeROI.width;
    maxSize.height = oSizeROI.height;

    rppHandle_t handle;
    hipStream_t stream;
    hipStreamCreate(&stream);
    rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);
    
    RppStatus status;
    //status = rppi_add_u8_pkd3_batchPD_gpu(d_input, d_input_second, srcSize, maxSize, d_output, noOfImages, handle);
    status = rppi_add_u8_pln1_batchPD_gpu((RppPtr_t)pSrc1, (RppPtr_t)pSrc2, srcSize, maxSize, (RppPtr_t)pDst, noOfImages, handle);

    hipDeviceSynchronize();

    rppDestroyGPU(handle);
    free(srcSize);

    return(hipRppStatusTocudaNppStatus(status));
}
NppStatus nppiAdd_8u_C1RSfs(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst, 
			    int nDstStep, NppiSize oSizeROI, int nScaleFactor)
{
    int noOfImages = 1;
    int ip_channel = 1;
    RppiSize *srcSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize));
    RppiSize maxSize;
    srcSize->width  = oSizeROI.width;
    srcSize->height = oSizeROI.height;
    maxSize.width  = oSizeROI.width;
    maxSize.height = oSizeROI.height;
        
    rppHandle_t handle;
    hipStream_t stream;
    hipStreamCreate(&stream);
    rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);
        
    RppStatus status;
    //status = rppi_add_u8_pkd3_batchPD_gpu(d_input, d_input_second, srcSize, maxSize, d_output, noOfImages, handle);
    status = rppi_add_u8_pln1_batchPD_gpu((RppPtr_t)pSrc1, (RppPtr_t)pSrc2, srcSize, maxSize, (RppPtr_t)pDst, noOfImages, handle);
        
    hipDeviceSynchronize();
        
    rppDestroyGPU(handle);
    free(srcSize);
        
    return(hipRppStatusTocudaNppStatus(status));
}
NppStatus nppiAdd_8u_C1IRSfs_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
				 int nScaleFactor, NppStreamContext nppStreamCtx)
{
    int noOfImages = 1;
    int ip_channel = 1;
    RppiSize *srcSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize));
    RppiSize maxSize;
    srcSize->width  = oSizeROI.width;
    srcSize->height = oSizeROI.height;
    maxSize.width  = oSizeROI.width;
    maxSize.height = oSizeROI.height;
	unsigned long long oBufferSize = 0;
	oBufferSize = (unsigned long long)maxSize.height * (unsigned long long)maxSize.width * (unsigned long long)ip_channel * (unsigned long long)noOfImages;
	int *pDst;
	hipMalloc(&pDst, oBufferSize * sizeof(Rpp8u));

    rppHandle_t handle;
    hipStream_t stream;
    hipStreamCreate(&stream);
    rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

    RppStatus status;
    //status = rppi_add_u8_pkd3_batchPD_gpu(d_input, d_input_second, srcSize, maxSize, d_output, noOfImages, handle);
    status = rppi_add_u8_pln1_batchPD_gpu((RppPtr_t)pSrc, (RppPtr_t)pSrcDst, srcSize, maxSize, (RppPtr_t)pDst, noOfImages, handle);

	hipMemcpy(pSrcDst, pDst, oBufferSize * sizeof(Rpp8u), hipMemcpyDeviceToDevice);
    hipDeviceSynchronize();

    rppDestroyGPU(handle);
    free(srcSize);
	hipFree(pDst);

    return(hipRppStatusTocudaNppStatus(status));
}
NppStatus nppiAdd_8u_C1IRSfs(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor)
{
    int noOfImages = 1;
    int ip_channel = 1;
    RppiSize *srcSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize));
    RppiSize maxSize;
    srcSize->width  = oSizeROI.width;
    srcSize->height = oSizeROI.height;
    maxSize.width  = oSizeROI.width;
    maxSize.height = oSizeROI.height;
	unsigned long long oBufferSize = 0;
    oBufferSize = (unsigned long long)maxSize.height * (unsigned long long)maxSize.width * (unsigned long long)ip_channel * (unsigned long long)noOfImages;
    int *pDst;
    hipMalloc(&pDst, oBufferSize * sizeof(Rpp8u));

    rppHandle_t handle;
    hipStream_t stream;
    hipStreamCreate(&stream);
    rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

    RppStatus status;
    //status = rppi_add_u8_pkd3_batchPD_gpu(d_input, d_input_second, srcSize, maxSize, d_output, noOfImages, handle);
    status = rppi_add_u8_pln1_batchPD_gpu((RppPtr_t)pSrc, (RppPtr_t)pSrcDst, srcSize, maxSize, (RppPtr_t)pDst, noOfImages, handle);

    hipMemcpy(pSrcDst, pDst, oBufferSize * sizeof(Rpp8u), hipMemcpyDeviceToDevice);
    hipDeviceSynchronize();

    rppDestroyGPU(handle);
    free(srcSize);
	hipFree(pDst);

    return(hipRppStatusTocudaNppStatus(status));
}
NppStatus nppiAdd_8u_C3RSfs_Ctx(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst, 
				int nDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx)
{
    int noOfImages = 1;
    int ip_channel = 3;
    RppiSize *srcSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize));
    RppiSize maxSize;
    srcSize->width  = oSizeROI.width;
    srcSize->height = oSizeROI.height;
    maxSize.width  = oSizeROI.width;
    maxSize.height = oSizeROI.height;

    rppHandle_t handle;
    hipStream_t stream;
    hipStreamCreate(&stream);
    rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

    RppStatus status;
    //status = rppi_add_u8_pkd3_batchPD_gpu(d_input, d_input_second, srcSize, maxSize, d_output, noOfImages, handle);
    status = rppi_add_u8_pkd3_batchPD_gpu((RppPtr_t)pSrc1, (RppPtr_t)pSrc2, srcSize, maxSize, (RppPtr_t)pDst, noOfImages, handle);

    hipDeviceSynchronize();

    rppDestroyGPU(handle);
    free(srcSize);

    return(hipRppStatusTocudaNppStatus(status));
}
NppStatus nppiAdd_8u_C3IRSfs_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx)
{
    int noOfImages = 1;
    int ip_channel = 3;
    RppiSize *srcSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize));
    RppiSize maxSize;
    srcSize->width  = oSizeROI.width;
    srcSize->height = oSizeROI.height;
    maxSize.width  = oSizeROI.width;
    maxSize.height = oSizeROI.height;
	unsigned long long oBufferSize = 0;
    oBufferSize = (unsigned long long)maxSize.height * (unsigned long long)maxSize.width * (unsigned long long)ip_channel * (unsigned long long)noOfImages;
    int *pDst;
    hipMalloc(&pDst, oBufferSize * sizeof(Rpp8u));
	
    rppHandle_t handle;
    hipStream_t stream;
    hipStreamCreate(&stream);
    rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

    RppStatus status;
    //status = rppi_add_u8_pkd3_batchPD_gpu(d_input, d_input_second, srcSize, maxSize, d_output, noOfImages, handle);
    status = rppi_add_u8_pkd3_batchPD_gpu((RppPtr_t)pSrc, (RppPtr_t)pSrcDst, srcSize, maxSize, (RppPtr_t)pDst, noOfImages, handle);

    hipMemcpy(pSrcDst, pDst, oBufferSize * sizeof(Rpp8u), hipMemcpyDeviceToDevice);
    hipDeviceSynchronize();

    rppDestroyGPU(handle);
    free(srcSize);
	hipFree(pDst);

    return(hipRppStatusTocudaNppStatus(status));
}
NppStatus nppiAdd_8u_C3IRSfs(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor)
{
    int noOfImages = 1;
    int ip_channel = 3;
    RppiSize *srcSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize));
    RppiSize maxSize;
    srcSize->width  = oSizeROI.width;
    srcSize->height = oSizeROI.height;
    maxSize.width  = oSizeROI.width; 
    maxSize.height = oSizeROI.height;
	unsigned long long oBufferSize = 0;
    oBufferSize = (unsigned long long)maxSize.height * (unsigned long long)maxSize.width * (unsigned long long)ip_channel * (unsigned long long)noOfImages;
    int *pDst;
    hipMalloc(&pDst, oBufferSize * sizeof(Rpp8u));
        
    rppHandle_t handle;
    hipStream_t stream;
    hipStreamCreate(&stream);
    rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);
        
    RppStatus status;
    //status = rppi_add_u8_pkd3_batchPD_gpu(d_input, d_input_second, srcSize, maxSize, d_output, noOfImages, handle);
    status = rppi_add_u8_pkd3_batchPD_gpu((RppPtr_t)pSrc, (RppPtr_t)pSrcDst, srcSize, maxSize, (RppPtr_t)pDst, noOfImages, handle);
        
    hipMemcpy(pSrcDst, pDst, oBufferSize * sizeof(Rpp8u), hipMemcpyDeviceToDevice);
    hipDeviceSynchronize();

    rppDestroyGPU(handle);
    free(srcSize);
	hipFree(pDst);	

    return(hipRppStatusTocudaNppStatus(status));
}

/*NppStatus nppiAdd_16f_C3RSfs(const Npp16f * pSrc1, int nSrc1Step, const Npp16f * pSrc2, int nSrc2Step,
                        Npp16f * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor)

{
        int noOfImages = 1;
        int ip_channel = 3;
        RppiSize *srcSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize));
        RppiSize maxSize;
        srcSize->width  = oSizeROI.width;
        srcSize->height = oSizeROI.height;
        maxSize.width  = oSizeROI.width;
        maxSize.height = oSizeROI.height;

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        //status = rppi_add_u8_pkd3_batchPD_gpu(d_input, d_input_second, srcSize, maxSize, d_output, noOfImages, handle);
        status = rppi_add_16f_pkd3_batchPD_gpu((RppPtr_t)pSrc1, (RppPtr_t)pSrc2, srcSize, maxSize, (RppPtr_t)pDst, noOfImages, handle);

        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        free(srcSize);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiAdd_16s_C3RSfs(const Npp16s * pSrc1, int nSrc1Step, const Npp16s * pSrc2, int nSrc2Step,
                        Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor)

{
        int noOfImages = 1;
        int ip_channel = 3;
        RppiSize *srcSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize));
        RppiSize maxSize; 
        srcSize->width  = oSizeROI.width;
        srcSize->height = oSizeROI.height;
        maxSize.width  = oSizeROI.width;
        maxSize.height = oSizeROI.height;
        
        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        //status = rppi_add_u8_pkd3_batchPD_gpu(d_input, d_input_second, srcSize, maxSize, d_output, noOfImages, handle);
        status = rppi_add_16s_pkd3_batchPD_gpu((RppPtr_t)pSrc1, (RppPtr_t)pSrc2, srcSize, maxSize, (RppPtr_t)pDst, noOfImages, handle);

        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        free(srcSize);

        return(hipRppStatusTocudaNppStatus(status));
}*/

NppStatus nppiAdd_32f_C3RSfs(const Npp32f * pSrc1, int nSrc1Step, const Npp32f * pSrc2, int nSrc2Step,
                        Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor)

{
    int noOfImages = 1;
    int ip_channel = 3;
    RppiSize *srcSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize));
    RppiSize maxSize; 
    srcSize->width  = oSizeROI.width;
    srcSize->height = oSizeROI.height;
    maxSize.width  = oSizeROI.width;
    maxSize.height = oSizeROI.height;
        
    rppHandle_t handle;
    hipStream_t stream;
    hipStreamCreate(&stream);
    rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

    RppStatus status;
    //status = rppi_add_u8_pkd3_batchPD_gpu(d_input, d_input_second, srcSize, maxSize, d_output, noOfImages, handle);
    status = rppi_add_32f_pkd3_batchPD_gpu((RppPtr_t)pSrc1, (RppPtr_t)pSrc2, srcSize, maxSize, (RppPtr_t)pDst, noOfImages, handle);

    hipDeviceSynchronize();

    rppDestroyGPU(handle);
    free(srcSize);

    return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiAddSquare_8u32f_C1IR(const Npp8u *pSrc, int nSrcStep, Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI)
{
    int noOfImages = 1;
    int ip_channel = 1;
	unsigned long long oBufferSize = 0;
    RppiSize *srcSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize));
    RppiSize maxSize;
    srcSize->width  = oSizeROI.width;
    srcSize->height = oSizeROI.height;
    maxSize.width  = oSizeROI.width;
    maxSize.height = oSizeROI.height;
	oBufferSize = (unsigned long long)maxSize.height * (unsigned long long)maxSize.width * (unsigned long long)ip_channel * (unsigned long long)noOfImages;

    rppHandle_t handle;
    hipStream_t stream;
    hipStreamCreate(&stream);
    rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

    RppStatus status;
    //status = rppi_add_u8_pkd3_batchPD_gpu(d_input, d_input_second, srcSize, maxSize, d_output, noOfImages, handle);
    status = rppi_accumulate_squared_u8_pln1_batchPD_gpu((RppPtr_t)pSrc, srcSize, maxSize, noOfImages, handle);
	hipMemcpy(pSrcDst, pSrc, oBufferSize * sizeof(Rpp8u), hipMemcpyDeviceToDevice);
    hipDeviceSynchronize();

    rppDestroyGPU(handle);
    free(srcSize);

    return(hipRppStatusTocudaNppStatus(status));
}
NppStatus nppiAddSquare_8u32f_C1IR_Ctx(const Npp8u *pSrc, int nSrcStep, Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx)
{
    int noOfImages = 1;
    int ip_channel = 1;
    unsigned long long oBufferSize = 0;
    RppiSize *srcSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize));
    RppiSize maxSize;
    srcSize->width  = oSizeROI.width;
    srcSize->height = oSizeROI.height;
    maxSize.width  = oSizeROI.width;
    maxSize.height = oSizeROI.height;
    oBufferSize = (unsigned long long)maxSize.height * (unsigned long long)maxSize.width * (unsigned long long)ip_channel * (unsigned long long)noOfImages;

    rppHandle_t handle;
    hipStream_t stream;
    hipStreamCreate(&stream);
    rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

    RppStatus status;
    //status = rppi_add_u8_pkd3_batchPD_gpu(d_input, d_input_second, srcSize, maxSize, d_output, noOfImages, handle);
    status = rppi_accumulate_squared_u8_pln1_batchPD_gpu((RppPtr_t)pSrc, srcSize, maxSize, noOfImages, handle);
    hipMemcpy(pSrcDst, pSrc, oBufferSize * sizeof(Rpp8u), hipMemcpyDeviceToDevice);
    hipDeviceSynchronize();

    rppDestroyGPU(handle);
    free(srcSize);

    return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiAddWeighted_8u32f_C1IR(const Npp8u *pSrc, int nSrcStep, Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, Npp32f nAlpha)
{
    int noOfImages = 1;
    int ip_channel = 1;
    unsigned long long oBufferSize = 0;
    RppiSize *srcSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize));
    RppiSize maxSize;
    srcSize->width  = oSizeROI.width;
    srcSize->height = oSizeROI.height;
    maxSize.width  = oSizeROI.width;
    maxSize.height = oSizeROI.height;
    Rpp32f alpha[1] = {0};
    alpha[0] = nAlpha;
    oBufferSize = (unsigned long long)maxSize.height * (unsigned long long)maxSize.width * (unsigned long long)ip_channel * (unsigned long long)noOfImages;

    rppHandle_t handle;
    hipStream_t stream;
    hipStreamCreate(&stream);
    rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

    RppStatus status;
    //status = rppi_add_u8_pkd3_batchPD_gpu(d_input, d_input_second, srcSize, maxSize, d_output, noOfImages, handle);
    status = rppi_accumulate_weighted_u8_pln1_batchPD_gpu((RppPtr_t)pSrcDst, (RppPtr_t)pSrc, srcSize, maxSize, alpha, noOfImages, handle);
 
    hipDeviceSynchronize();

    rppDestroyGPU(handle);
    free(srcSize);

    return(hipRppStatusTocudaNppStatus(status));
}
NppStatus nppiAddWeighted_8u32f_C1IR_Ctx(const Npp8u *pSrc, int nSrcStep, Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, Npp32f nAlpha, NppStreamContext nppStreamCtx)
{
    int noOfImages = 1;
    int ip_channel = 1;
    unsigned long long oBufferSize = 0;
    RppiSize *srcSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize));
    RppiSize maxSize;
    srcSize->width  = oSizeROI.width;
    srcSize->height = oSizeROI.height;
    maxSize.width  = oSizeROI.width;
    maxSize.height = oSizeROI.height;
    Rpp32f alpha[1] = {0};
    alpha[0] = nAlpha;
    oBufferSize = (unsigned long long)maxSize.height * (unsigned long long)maxSize.width * (unsigned long long)ip_channel * (unsigned long long)noOfImages;
         
    rppHandle_t handle;
    hipStream_t stream;
    hipStreamCreate(&stream);
    rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

    RppStatus status;
    //status = rppi_add_u8_pkd3_batchPD_gpu(d_input, d_input_second, srcSize, maxSize, d_output, noOfImages, handle);
    status = rppi_accumulate_weighted_u8_pln1_batchPD_gpu((RppPtr_t)pSrcDst, (RppPtr_t)pSrc, srcSize, maxSize, alpha, noOfImages, handle);

    hipDeviceSynchronize();

    rppDestroyGPU(handle);
    free(srcSize);

    return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiMul_8u_C3RSfs(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor)
{
    int noOfImages = 1;
    int ip_channel = 3;
    //unsigned long long oBufferSize = 0;
    RppiSize *srcSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize));
    RppiSize maxSize;
    srcSize->width  = oSizeROI.width;
    srcSize->height = oSizeROI.height;
    maxSize.width  = oSizeROI.width;
    maxSize.height = oSizeROI.height;
    //oBufferSize = (unsigned long long)maxSize.height * (unsigned long long)maxSize.width * (unsigned long long)ip_channel * (unsigned long long)noOfImages;

    rppHandle_t handle;
    hipStream_t stream;
    hipStreamCreate(&stream);
    rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

    RppStatus status;
    //status = rppi_add_u8_pkd3_batchPD_gpu(d_input, d_input_second, srcSize, maxSize, d_output, noOfImages, handle);
    status = rppi_multiply_u8_pkd3_batchPD_gpu((RppPtr_t)pSrc1, (RppPtr_t)pSrc2, srcSize, maxSize, (RppPtr_t)pDst, noOfImages, handle);
    //hipMemcpy(pSrcDst, pSrc, oBufferSize * sizeof(Rpp8u), hipMemcpyDeviceToDevice);
    hipDeviceSynchronize();

    rppDestroyGPU(handle);
    free(srcSize);

    return(hipRppStatusTocudaNppStatus(status));
}
NppStatus nppiMul_8u_C1RSfs_Ctx(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx)
{
        int noOfImages = 1;
        int ip_channel = 1;
        //unsigned long long oBufferSize = 0;
        RppiSize *srcSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize));
        RppiSize maxSize;
        srcSize->width  = oSizeROI.width;
        srcSize->height = oSizeROI.height;
        maxSize.width  = oSizeROI.width;
        maxSize.height = oSizeROI.height;
        //oBufferSize = (unsigned long long)maxSize.height * (unsigned long long)maxSize.width * (unsigned long long)ip_channel * (unsigned long long)noOfImages;

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        //status = rppi_add_u8_pkd3_batchPD_gpu(d_input, d_input_second, srcSize, maxSize, d_output, noOfImages, handle);
        status = rppi_multiply_u8_pln1_batchPD_gpu((RppPtr_t)pSrc1, (RppPtr_t)pSrc2, srcSize, maxSize, (RppPtr_t)pDst, noOfImages, handle);
        //hipMemcpy(pSrcDst, pSrc, oBufferSize * sizeof(Rpp8u), hipMemcpyDeviceToDevice);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        free(srcSize);
        //hipFree(d_input);
        //hipFree(d_input_second);
        //hipFree(d_output);

        return(hipRppStatusTocudaNppStatus(status));
}
NppStatus nppiMul_8u_C1RSfs(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor)
{
        int noOfImages = 1;
        int ip_channel = 1;
        //unsigned long long oBufferSize = 0;
        RppiSize *srcSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize));
        RppiSize maxSize;
        srcSize->width  = oSizeROI.width;
        srcSize->height = oSizeROI.height;
        maxSize.width  = oSizeROI.width;
        maxSize.height = oSizeROI.height;
        //oBufferSize = (unsigned long long)maxSize.height * (unsigned long long)maxSize.width * (unsigned long long)ip_channel * (unsigned long long)noOfImages;

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        //status = rppi_add_u8_pkd3_batchPD_gpu(d_input, d_input_second, srcSize, maxSize, d_output, noOfImages, handle);
        status = rppi_multiply_u8_pln1_batchPD_gpu((RppPtr_t)pSrc1, (RppPtr_t)pSrc2, srcSize, maxSize, (RppPtr_t)pDst, noOfImages, handle);
        //hipMemcpy(pSrcDst, pSrc, oBufferSize * sizeof(Rpp8u), hipMemcpyDeviceToDevice);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        free(srcSize);
        //hipFree(d_input);
        //hipFree(d_input_second);
        //hipFree(d_output);

        return(hipRppStatusTocudaNppStatus(status));
}
NppStatus nppiMul_8u_C1IRSfs_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx)
{
        int noOfImages = 1;
        int ip_channel = 1;
        unsigned long long oBufferSize = 0;
        RppiSize *srcSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize));
        RppiSize maxSize;
        srcSize->width  = oSizeROI.width;
        srcSize->height = oSizeROI.height;
        maxSize.width  = oSizeROI.width;
        maxSize.height = oSizeROI.height;
        oBufferSize = (unsigned long long)maxSize.height * (unsigned long long)maxSize.width * (unsigned long long)ip_channel * (unsigned long long)noOfImages;
	    int *pDst;
        hipMalloc(&pDst, oBufferSize * sizeof(Rpp8u));

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        //status = rppi_add_u8_pkd3_batchPD_gpu(d_input, d_input_second, srcSize, maxSize, d_output, noOfImages, handle);
        status = rppi_multiply_u8_pln1_batchPD_gpu((RppPtr_t)pSrc, (RppPtr_t)pSrcDst, srcSize, maxSize, (RppPtr_t)pDst, noOfImages, handle);
        hipMemcpy(pSrcDst, pDst, oBufferSize * sizeof(Rpp8u), hipMemcpyDeviceToDevice);
        hipDeviceSynchronize();


        rppDestroyGPU(handle);
        free(srcSize);
	    hipFree(pDst);
        //hipFree(d_input);
        //hipFree(d_input_second);
        //hipFree(d_output);

        return(hipRppStatusTocudaNppStatus(status));
}
NppStatus nppiMul_8u_C1IRSfs(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor)
{
        int noOfImages = 1;
        int ip_channel = 1;
        unsigned long long oBufferSize = 0;
        RppiSize *srcSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize));
        RppiSize maxSize;
        srcSize->width  = oSizeROI.width;
        srcSize->height = oSizeROI.height;
        maxSize.width  = oSizeROI.width;
        maxSize.height = oSizeROI.height;
        oBufferSize = (unsigned long long)maxSize.height * (unsigned long long)maxSize.width * (unsigned long long)ip_channel * (unsigned long long)noOfImages;
	int *pDst;
        hipMalloc(&pDst, oBufferSize * sizeof(Rpp8u));

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        //status = rppi_add_u8_pkd3_batchPD_gpu(d_input, d_input_second, srcSize, maxSize, d_output, noOfImages, handle);
        status = rppi_multiply_u8_pln1_batchPD_gpu((RppPtr_t)pSrc, (RppPtr_t)pSrcDst, srcSize, maxSize, (RppPtr_t)pDst, noOfImages, handle);
        hipMemcpy(pSrcDst, pDst, oBufferSize * sizeof(Rpp8u), hipMemcpyDeviceToDevice);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        free(srcSize);
	    hipFree(pDst);
        //hipFree(d_input);
        //hipFree(d_input_second);
        //hipFree(d_output);

        return(hipRppStatusTocudaNppStatus(status));
}
NppStatus nppiMul_8u_C3RSfs_Ctx(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx)
{
        int noOfImages = 1;
        int ip_channel = 3;
        //unsigned long long oBufferSize = 0;
        RppiSize *srcSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize));
        RppiSize maxSize;
        srcSize->width  = oSizeROI.width;
        srcSize->height = oSizeROI.height;
        maxSize.width  = oSizeROI.width;
        maxSize.height = oSizeROI.height;
        //oBufferSize = (unsigned long long)maxSize.height * (unsigned long long)maxSize.width * (unsigned long long)ip_channel * (unsigned long long)noOfImages;

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        //status = rppi_add_u8_pkd3_batchPD_gpu(d_input, d_input_second, srcSize, maxSize, d_output, noOfImages, handle);
        status = rppi_multiply_u8_pkd3_batchPD_gpu((RppPtr_t)pSrc1, (RppPtr_t)pSrc2, srcSize, maxSize, (RppPtr_t)pDst, noOfImages, handle);
        //hipMemcpy(pSrcDst, pSrc, oBufferSize * sizeof(Rpp8u), hipMemcpyDeviceToDevice);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        free(srcSize);
        //hipFree(d_input);
        //hipFree(d_input_second);
        //hipFree(d_output);

        return(hipRppStatusTocudaNppStatus(status));
}
NppStatus nppiMul_8u_C3IRSfs_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx)
{
        int noOfImages = 1;
        int ip_channel = 3;
        unsigned long long oBufferSize = 0;
	    int *pDst;
        RppiSize *srcSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize));
        RppiSize maxSize;
        srcSize->width  = oSizeROI.width;
        srcSize->height = oSizeROI.height;
        maxSize.width  = oSizeROI.width;
        maxSize.height = oSizeROI.height;
        oBufferSize = (unsigned long long)maxSize.height * (unsigned long long)maxSize.width * (unsigned long long)ip_channel * (unsigned long long)noOfImages;
	    hipMalloc(&pDst, oBufferSize * sizeof(Rpp8u));

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        //status = rppi_add_u8_pkd3_batchPD_gpu(d_input, d_input_second, srcSize, maxSize, d_output, noOfImages, handle);
        status = rppi_multiply_u8_pkd3_batchPD_gpu((RppPtr_t)pSrc, (RppPtr_t)pSrcDst, srcSize, maxSize, (RppPtr_t)pDst, noOfImages, handle);
        hipMemcpy(pSrcDst, pDst, oBufferSize * sizeof(Rpp8u), hipMemcpyDeviceToDevice);
        hipDeviceSynchronize();
      
        rppDestroyGPU(handle);
        free(srcSize);
	    hipFree(pDst);
        //hipFree(d_input);
        //hipFree(d_input_second);
        //hipFree(d_output);

        return(hipRppStatusTocudaNppStatus(status));
}
NppStatus nppiMul_8u_C3IRSfs(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor)
{
        int noOfImages = 1;
        int ip_channel = 3;
        unsigned long long oBufferSize = 0;
        int *pDst;
        RppiSize *srcSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize));
        RppiSize maxSize;
        srcSize->width  = oSizeROI.width;
        srcSize->height = oSizeROI.height;
        maxSize.width  = oSizeROI.width;
        maxSize.height = oSizeROI.height;
        oBufferSize = (unsigned long long)maxSize.height * (unsigned long long)maxSize.width * (unsigned long long)ip_channel * (unsigned long long)noOfImages;
        hipMalloc(&pDst, oBufferSize * sizeof(Rpp8u));

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        //status = rppi_add_u8_pkd3_batchPD_gpu(d_input, d_input_second, srcSize, maxSize, d_output, noOfImages, handle);
        status = rppi_multiply_u8_pkd3_batchPD_gpu((RppPtr_t)pSrc, (RppPtr_t)pSrcDst, srcSize, maxSize, (RppPtr_t)pDst, noOfImages, handle);
        hipMemcpy(pSrcDst, pDst, oBufferSize * sizeof(Rpp8u), hipMemcpyDeviceToDevice);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        free(srcSize);
        hipFree(pDst);
        //hipFree(d_input);
        //hipFree(d_input_second);
        //hipFree(d_output);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiSub_8u_C3RSfs(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor)
{
        int noOfImages = 1;
        int ip_channel = 3;
        //unsigned long long oBufferSize = 0;
        RppiSize *srcSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize));
        RppiSize maxSize;
        srcSize->width  = oSizeROI.width;
        srcSize->height = oSizeROI.height;
        maxSize.width  = oSizeROI.width;
        maxSize.height = oSizeROI.height;
        //oBufferSize = (unsigned long long)maxSize.height * (unsigned long long)maxSize.width * (unsigned long long)ip_channel * (unsigned long long)noOfImages;

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        //status = rppi_add_u8_pkd3_batchPD_gpu(d_input, d_input_second, srcSize, maxSize, d_output, noOfImages, handle);
        status = rppi_subtract_u8_pkd3_batchPD_gpu((RppPtr_t)pSrc1, (RppPtr_t)pSrc2, srcSize, maxSize, (RppPtr_t)pDst, noOfImages, handle);
        //hipMemcpy(pSrcDst, pSrc, oBufferSize * sizeof(Rpp8u), hipMemcpyDeviceToDevice);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        free(srcSize);
        //hipFree(d_input);
        //hipFree(d_input_second);
        //hipFree(d_output);

        return(hipRppStatusTocudaNppStatus(status));
}
NppStatus nppiSub_8u_C1RSfs_Ctx(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx)
{
        int noOfImages = 1;
        int ip_channel = 1;
        //unsigned long long oBufferSize = 0;
        RppiSize *srcSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize));
        RppiSize maxSize;
        srcSize->width  = oSizeROI.width;
        srcSize->height = oSizeROI.height;
        maxSize.width  = oSizeROI.width;
        maxSize.height = oSizeROI.height;
        //oBufferSize = (unsigned long long)maxSize.height * (unsigned long long)maxSize.width * (unsigned long long)ip_channel * (unsigned long long)noOfImages;

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        //status = rppi_add_u8_pkd3_batchPD_gpu(d_input, d_input_second, srcSize, maxSize, d_output, noOfImages, handle);
        status = rppi_subtract_u8_pln1_batchPD_gpu((RppPtr_t)pSrc1, (RppPtr_t)pSrc2, srcSize, maxSize, (RppPtr_t)pDst, noOfImages, handle);
        //hipMemcpy(pSrcDst, pSrc, oBufferSize * sizeof(Rpp8u), hipMemcpyDeviceToDevice);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        free(srcSize);
        //hipFree(d_input);
        //hipFree(d_input_second);
        //hipFree(d_output);

        return(hipRppStatusTocudaNppStatus(status));
}
NppStatus nppiSub_8u_C1RSfs(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor)
{
        int noOfImages = 1;
        int ip_channel = 1;
        //unsigned long long oBufferSize = 0;
        RppiSize *srcSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize));
        RppiSize maxSize;
        srcSize->width  = oSizeROI.width;
        srcSize->height = oSizeROI.height;
        maxSize.width  = oSizeROI.width;
        maxSize.height = oSizeROI.height;
        //oBufferSize = (unsigned long long)maxSize.height * (unsigned long long)maxSize.width * (unsigned long long)ip_channel * (unsigned long long)noOfImages;

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        //status = rppi_add_u8_pkd3_batchPD_gpu(d_input, d_input_second, srcSize, maxSize, d_output, noOfImages, handle);
        status = rppi_subtract_u8_pln1_batchPD_gpu((RppPtr_t)pSrc1, (RppPtr_t)pSrc2, srcSize, maxSize, (RppPtr_t)pDst, noOfImages, handle);
        //hipMemcpy(pSrcDst, pSrc, oBufferSize * sizeof(Rpp8u), hipMemcpyDeviceToDevice);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        free(srcSize);
        //hipFree(d_input);
        //hipFree(d_input_second);
        //hipFree(d_output);

        return(hipRppStatusTocudaNppStatus(status));
}
NppStatus nppiSub_8u_C1IRSfs_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx)
{
        int noOfImages = 1;
        int ip_channel = 1;
        unsigned long long oBufferSize = 0;
        RppiSize *srcSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize));
        RppiSize maxSize;
        srcSize->width  = oSizeROI.width;
        srcSize->height = oSizeROI.height;
        maxSize.width  = oSizeROI.width;
        maxSize.height = oSizeROI.height;
        oBufferSize = (unsigned long long)maxSize.height * (unsigned long long)maxSize.width * (unsigned long long)ip_channel * (unsigned long long)noOfImages;
	    int *pDst;
	    hipMalloc(&pDst, oBufferSize * sizeof(Rpp8u));

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        //status = rppi_add_u8_pkd3_batchPD_gpu(d_input, d_input_second, srcSize, maxSize, d_output, noOfImages, handle);
        status = rppi_subtract_u8_pln1_batchPD_gpu((RppPtr_t)pSrc, (RppPtr_t)pSrcDst, srcSize, maxSize, (RppPtr_t)pDst, noOfImages, handle);
        hipMemcpy(pSrcDst, pDst, oBufferSize * sizeof(Rpp8u), hipMemcpyDeviceToDevice);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        free(srcSize);
	    hipFree(pDst);
        //hipFree(d_input);
        //hipFree(d_input_second);
        //hipFree(d_output);

        return(hipRppStatusTocudaNppStatus(status));
}
NppStatus nppiSub_8u_C1IRSfs(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor)
{
        int noOfImages = 1;
        int ip_channel = 1;
        unsigned long long oBufferSize = 0;
        RppiSize *srcSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize)); 
        RppiSize maxSize;
        srcSize->width  = oSizeROI.width;
        srcSize->height = oSizeROI.height;
        maxSize.width  = oSizeROI.width;
        maxSize.height = oSizeROI.height;
        oBufferSize = (unsigned long long)maxSize.height * (unsigned long long)maxSize.width * (unsigned long long)ip_channel * (unsigned long long)noOfImages;
        int *pDst;
        hipMalloc(&pDst, oBufferSize * sizeof(Rpp8u));

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        //status = rppi_add_u8_pkd3_batchPD_gpu(d_input, d_input_second, srcSize, maxSize, d_output, noOfImages, handle);
        status = rppi_subtract_u8_pln1_batchPD_gpu((RppPtr_t)pSrc, (RppPtr_t)pSrcDst, srcSize, maxSize, (RppPtr_t)pDst, noOfImages, handle);
        hipMemcpy(pSrcDst, pDst, oBufferSize * sizeof(Rpp8u), hipMemcpyDeviceToDevice);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        free(srcSize);
        hipFree(pDst);
        //hipFree(d_input);
        //hipFree(d_input_second);
        //hipFree(d_output);

        return(hipRppStatusTocudaNppStatus(status));
}
NppStatus nppiSub_8u_C3RSfs_Ctx(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx)
{
        int noOfImages = 1;
        int ip_channel = 3;
        //unsigned long long oBufferSize = 0;
        RppiSize *srcSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize));
        RppiSize maxSize;
        srcSize->width  = oSizeROI.width;
        srcSize->height = oSizeROI.height;
        maxSize.width  = oSizeROI.width;
        maxSize.height = oSizeROI.height;
        //oBufferSize = (unsigned long long)maxSize.height * (unsigned long long)maxSize.width * (unsigned long long)ip_channel * (unsigned long long)noOfImages;

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        //status = rppi_add_u8_pkd3_batchPD_gpu(d_input, d_input_second, srcSize, maxSize, d_output, noOfImages, handle);
        status = rppi_subtract_u8_pkd3_batchPD_gpu((RppPtr_t)pSrc1, (RppPtr_t)pSrc2, srcSize, maxSize, (RppPtr_t)pDst, noOfImages, handle);
        //hipMemcpy(pSrcDst, pSrc, oBufferSize * sizeof(Rpp8u), hipMemcpyDeviceToDevice);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        free(srcSize);
        //hipFree(d_input);
        //hipFree(d_input_second);
        //hipFree(d_output);

        return(hipRppStatusTocudaNppStatus(status));
}
NppStatus nppiSub_8u_C3IRSfs_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx)
{
        int noOfImages = 1;
        int ip_channel = 3;
        unsigned long long oBufferSize = 0;
        RppiSize *srcSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize));
        RppiSize maxSize;
        srcSize->width  = oSizeROI.width;
        srcSize->height = oSizeROI.height;
        maxSize.width  = oSizeROI.width;
        maxSize.height = oSizeROI.height;
        oBufferSize = (unsigned long long)maxSize.height * (unsigned long long)maxSize.width * (unsigned long long)ip_channel * (unsigned long long)noOfImages;
	    int *pDst;
	    hipMalloc(&pDst, oBufferSize * sizeof(Rpp8u));

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        //status = rppi_add_u8_pkd3_batchPD_gpu(d_input, d_input_second, srcSize, maxSize, d_output, noOfImages, handle);
        status = rppi_subtract_u8_pkd3_batchPD_gpu((RppPtr_t)pSrc, (RppPtr_t)pSrcDst, srcSize, maxSize, (RppPtr_t)pDst, noOfImages, handle);
        hipMemcpy(pSrcDst, pDst, oBufferSize * sizeof(Rpp8u), hipMemcpyDeviceToDevice);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        free(srcSize);
	    hipFree(pDst);
        //hipFree(d_input);
        //hipFree(d_input_second);
        //hipFree(d_output);

        return(hipRppStatusTocudaNppStatus(status));
}
NppStatus nppiSub_8u_C3IRSfs(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor)
{
        int noOfImages = 1;
        int ip_channel = 3;
        unsigned long long oBufferSize = 0;
        RppiSize *srcSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize));
        RppiSize maxSize;
        srcSize->width  = oSizeROI.width;
        srcSize->height = oSizeROI.height;
        maxSize.width  = oSizeROI.width;
        maxSize.height = oSizeROI.height;
        oBufferSize = (unsigned long long)maxSize.height * (unsigned long long)maxSize.width * (unsigned long long)ip_channel * (unsigned long long)noOfImages;
        int *pDst;
        hipMalloc(&pDst, oBufferSize * sizeof(Rpp8u));

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        //status = rppi_add_u8_pkd3_batchPD_gpu(d_input, d_input_second, srcSize, maxSize, d_output, noOfImages, handle);
        status = rppi_subtract_u8_pkd3_batchPD_gpu((RppPtr_t)pSrc, (RppPtr_t)pSrcDst, srcSize, maxSize, (RppPtr_t)pDst, noOfImages, handle);
        hipMemcpy(pSrcDst, pDst, oBufferSize * sizeof(Rpp8u), hipMemcpyDeviceToDevice);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        free(srcSize);
        hipFree(pDst);
        //hipFree(d_input);
        //hipFree(d_input_second);
        //hipFree(d_output);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiAbsDiff_8u_C3R(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst, int nDstStep, NppiSize oSizeROI)
{
        int noOfImages = 1;
        int ip_channel = 3;
        //unsigned long long oBufferSize = 0;
        RppiSize *srcSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize));
        RppiSize maxSize;
        srcSize->width  = oSizeROI.width;
        srcSize->height = oSizeROI.height;
        maxSize.width  = oSizeROI.width;
        maxSize.height = oSizeROI.height;
        //oBufferSize = (unsigned long long)maxSize.height * (unsigned long long)maxSize.width * (unsigned long long)ip_channel * (unsigned long long)noOfImages;

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        //status = rppi_add_u8_pkd3_batchPD_gpu(d_input, d_input_second, srcSize, maxSize, d_output, noOfImages, handle);
        status = rppi_absolute_difference_u8_pkd3_batchPD_gpu((RppPtr_t)pSrc1, (RppPtr_t)pSrc2, srcSize, maxSize, (RppPtr_t)pDst, noOfImages, handle);
        //hipMemcpy(pSrcDst, pSrc, oBufferSize * sizeof(Rpp8u), hipMemcpyDeviceToDevice);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        free(srcSize);
        //hipFree(d_input);
        //hipFree(d_input_second);
        //hipFree(d_output);

        return(hipRppStatusTocudaNppStatus(status));
}
NppStatus nppiAbsDiff_8u_C1R_Ctx(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx)
{
        int noOfImages = 1;
        int ip_channel = 1;
        //unsigned long long oBufferSize = 0;
        RppiSize *srcSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize));
        RppiSize maxSize;
        srcSize->width  = oSizeROI.width;
        srcSize->height = oSizeROI.height;
        maxSize.width  = oSizeROI.width;
        maxSize.height = oSizeROI.height;
        //oBufferSize = (unsigned long long)maxSize.height * (unsigned long long)maxSize.width * (unsigned long long)ip_channel * (unsigned long long)noOfImages;

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        //status = rppi_add_u8_pkd3_batchPD_gpu(d_input, d_input_second, srcSize, maxSize, d_output, noOfImages, handle);
        status = rppi_absolute_difference_u8_pln1_batchPD_gpu((RppPtr_t)pSrc1, (RppPtr_t)pSrc2, srcSize, maxSize, (RppPtr_t)pDst, noOfImages, handle);
        //hipMemcpy(pSrcDst, pSrc, oBufferSize * sizeof(Rpp8u), hipMemcpyDeviceToDevice);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        free(srcSize);
        //hipFree(d_input);
        //hipFree(d_input_second);
        //hipFree(d_output);

        return(hipRppStatusTocudaNppStatus(status));
}
NppStatus nppiAbsDiff_8u_C1R(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst, int nDstStep, NppiSize oSizeROI)
{
        int noOfImages = 1;
        int ip_channel = 1;
        //unsigned long long oBufferSize = 0;
        RppiSize *srcSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize));
        RppiSize maxSize;
        srcSize->width  = oSizeROI.width;
        srcSize->height = oSizeROI.height;
        maxSize.width  = oSizeROI.width;
        maxSize.height = oSizeROI.height;
        //oBufferSize = (unsigned long long)maxSize.height * (unsigned long long)maxSize.width * (unsigned long long)ip_channel * (unsigned long long)noOfImages;

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        //status = rppi_add_u8_pkd3_batchPD_gpu(d_input, d_input_second, srcSize, maxSize, d_output, noOfImages, handle);
        status = rppi_absolute_difference_u8_pln1_batchPD_gpu((RppPtr_t)pSrc1, (RppPtr_t)pSrc2, srcSize, maxSize, (RppPtr_t)pDst, noOfImages, handle);
        //hipMemcpy(pSrcDst, pSrc, oBufferSize * sizeof(Rpp8u), hipMemcpyDeviceToDevice);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        free(srcSize);
        //hipFree(d_input);
        //hipFree(d_input_second);
        //hipFree(d_output);

        return(hipRppStatusTocudaNppStatus(status));
}
NppStatus nppiAbsDiff_8u_C3R_Ctx(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx)
{
        int noOfImages = 1;
        int ip_channel = 3;
        //unsigned long long oBufferSize = 0;
        RppiSize *srcSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize));
        RppiSize maxSize;
        srcSize->width  = oSizeROI.width;
        srcSize->height = oSizeROI.height;
        maxSize.width  = oSizeROI.width;
        maxSize.height = oSizeROI.height;
        //oBufferSize = (unsigned long long)maxSize.height * (unsigned long long)maxSize.width * (unsigned long long)ip_channel * (unsigned long long)noOfImages;

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        //status = rppi_add_u8_pkd3_batchPD_gpu(d_input, d_input_second, srcSize, maxSize, d_output, noOfImages, handle);
        status = rppi_absolute_difference_u8_pkd3_batchPD_gpu((RppPtr_t)pSrc1, (RppPtr_t)pSrc2, srcSize, maxSize, (RppPtr_t)pDst, noOfImages, handle);
        //hipMemcpy(pSrcDst, pSrc, oBufferSize * sizeof(Rpp8u), hipMemcpyDeviceToDevice);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        free(srcSize);
        //hipFree(d_input);
        //hipFree(d_input_second);
        //hipFree(d_output);

        return(hipRppStatusTocudaNppStatus(status));
}
/*NppStatus nppiAnd_8u_C3R(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst, int nDstStep, NppiSize oSizeROI)
{
        int noOfImages = 1;
        int ip_channel = 3;
        //unsigned long long oBufferSize = 0;
        RppiSize *srcSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize));
        RppiSize maxSize;
        srcSize->width  = oSizeROI.width;
        srcSize->height = oSizeROI.height;
        maxSize.width  = oSizeROI.width;
        maxSize.height = oSizeROI.height;
        //oBufferSize = (unsigned long long)maxSize.height * (unsigned long long)maxSize.width * (unsigned long long)ip_channel * (unsigned long long)noOfImages;

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);
        clock_t start, end;
        double gpu_time_used;

        RppStatus status;
        start = clock();
        //status = rppi_add_u8_pkd3_batchPD_gpu(d_input, d_input_second, srcSize, maxSize, d_output, noOfImages, handle);
        status = rppi_bitwise_AND_u8_pkd3_batchPD_gpu((RppPtr_t)pSrc1, (RppPtr_t)pSrc2, srcSize, maxSize, (RppPtr_t)pDst, noOfImages, handle);
        //hipMemcpy(pSrcDst, pSrc, oBufferSize * sizeof(Rpp8u), hipMemcpyDeviceToDevice);
        hipDeviceSynchronize();
        end = clock();

        printf("\nnppiAnd_8u_C3R is %d\n", status);
        gpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
        std::cout << "\nGPU Time - And : " << gpu_time_used;
        printf("\n");
        //hipMemcpy(pDst, d_output, oBufferSize * sizeof(Rpp8u), hipMemcpyDeviceToDevice);
        //sleep(10);
        rppDestroyGPU(handle);
        free(srcSize);
        //hipFree(d_input);
        //hipFree(d_input_second);
        //hipFree(d_output);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiNot_8u_C3R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI)
{
        int noOfImages = 1;
        int ip_channel = 3;
        //unsigned long long oBufferSize = 0;
        RppiSize *srcSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize));
        RppiSize maxSize;
        srcSize->width  = oSizeROI.width;
        srcSize->height = oSizeROI.height;
        maxSize.width  = oSizeROI.width;
        maxSize.height = oSizeROI.height;
        //oBufferSize = (unsigned long long)maxSize.height * (unsigned long long)maxSize.width * (unsigned long long)ip_channel * (unsigned long long)noOfImages;

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);
        clock_t start, end;
        double gpu_time_used;

        RppStatus status;
        start = clock();
        //status = rppi_add_u8_pkd3_batchPD_gpu(d_input, d_input_second, srcSize, maxSize, d_output, noOfImages, handle);
        status = rppi_bitwise_NOT_u8_pkd3_batchPD_gpu((RppPtr_t)pSrc, srcSize, maxSize, (RppPtr_t)pDst, noOfImages, handle);
        //hipMemcpy(pSrcDst, pSrc, oBufferSize * sizeof(Rpp8u), hipMemcpyDeviceToDevice);
        hipDeviceSynchronize();
        end = clock();

        printf("\nnppiNot_8u_C3R is %d\n", status);
        gpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
        std::cout << "\nGPU Time - Not : " << gpu_time_used;
        printf("\n");
        //hipMemcpy(pDst, d_output, oBufferSize * sizeof(Rpp8u), hipMemcpyDeviceToDevice);
        //sleep(10);
        rppDestroyGPU(handle);
        free(srcSize);
        //hipFree(d_input);
        //hipFree(d_input_second);
        //hipFree(d_output);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiXor_8u_C3R(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst, int nDstStep, NppiSize oSizeROI)
{
        int noOfImages = 1;
        int ip_channel = 3;
        //unsigned long long oBufferSize = 0;
        RppiSize *srcSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize));
        RppiSize maxSize;
        srcSize->width  = oSizeROI.width;
        srcSize->height = oSizeROI.height;
        maxSize.width  = oSizeROI.width;
        maxSize.height = oSizeROI.height;
        //oBufferSize = (unsigned long long)maxSize.height * (unsigned long long)maxSize.width * (unsigned long long)ip_channel * (unsigned long long)noOfImages;

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);
        clock_t start, end;
        double gpu_time_used;

        RppStatus status;
        start = clock();
        //status = rppi_add_u8_pkd3_batchPD_gpu(d_input, d_input_second, srcSize, maxSize, d_output, noOfImages, handle);
        status = rppi_exclusive_OR_u8_pkd3_batchPD_gpu((RppPtr_t)pSrc1, (RppPtr_t)pSrc2, srcSize, maxSize, (RppPtr_t)pDst, noOfImages, handle);
        //hipMemcpy(pSrcDst, pSrc, oBufferSize * sizeof(Rpp8u), hipMemcpyDeviceToDevice);
        hipDeviceSynchronize();
        end = clock();

        printf("\nnppiXor_8u_C3R is %d\n", status);
        gpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
        std::cout << "\nGPU Time - Xor : " << gpu_time_used;
        printf("\n");
        //hipMemcpy(pDst, d_output, oBufferSize * sizeof(Rpp8u), hipMemcpyDeviceToDevice);
        //sleep(10);
        rppDestroyGPU(handle);
        free(srcSize);
        //hipFree(d_input);
        //hipFree(d_input_second);
        //hipFree(d_output);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiOr_8u_C3R(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst, int nDstStep, NppiSize oSizeROI)
{
        int noOfImages = 1;
        int ip_channel = 3;
        //unsigned long long oBufferSize = 0;
        RppiSize *srcSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize));
        RppiSize maxSize;
        srcSize->width  = oSizeROI.width;
        srcSize->height = oSizeROI.height;
        maxSize.width  = oSizeROI.width;
        maxSize.height = oSizeROI.height;
        //oBufferSize = (unsigned long long)maxSize.height * (unsigned long long)maxSize.width * (unsigned long long)ip_channel * (unsigned long long)noOfImages;

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);
        clock_t start, end;
        double gpu_time_used;

        RppStatus status;
        start = clock();
        //status = rppi_add_u8_pkd3_batchPD_gpu(d_input, d_input_second, srcSize, maxSize, d_output, noOfImages, handle);
        status = rppi_inclusive_OR_u8_pkd3_batchPD_gpu((RppPtr_t)pSrc1, (RppPtr_t)pSrc2, srcSize, maxSize, (RppPtr_t)pDst, noOfImages, handle);
        //hipMemcpy(pSrcDst, pSrc, oBufferSize * sizeof(Rpp8u), hipMemcpyDeviceToDevice);
        hipDeviceSynchronize();
        end = clock();

        printf("\nnppiOr_8u_C3R is %d\n", status);
        gpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
        std::cout << "\nGPU Time - Or : " << gpu_time_used;
        printf("\n");
        //hipMemcpy(pDst, d_output, oBufferSize * sizeof(Rpp8u), hipMemcpyDeviceToDevice);
        //sleep(10);
        rppDestroyGPU(handle);
        free(srcSize);
        //hipFree(d_input);
        //hipFree(d_input_second);
        //hipFree(d_output);

        return(hipRppStatusTocudaNppStatus(status));
}*/

/*NppStatus nppiAdd_8u_C3RSfs_handle(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step,
                        Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor,  rppHandle_t rppHandle)

{
        int noOfImages = 1;
        RppiSize *srcSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize));
        RppiSize maxSize;
        srcSize->width  = oSizeROI.width;
        srcSize->height = oSizeROI.height;
        maxSize.width  = oSizeROI.width;
        maxSize.height = oSizeROI.height;

        printf("\nnppiAdd_8u_C3RSfs_handle srcSize.width is %d,srcSize.height is %d,maxSize.width is %d,maxSize.height is %d\n", srcSize->width,srcSize->height,maxSize.width,maxSize.height);

        RppStatus status;
        printf("\nrppi add sPtr1 is %p,sPtr2 is %p,dPtr is %p,Handle is %p\n", pSrc1,pSrc1,pDst,rppHandle);
        status = rppi_add_u8_pkd3_batchPD_gpu((RppPtr_t)pSrc1, (RppPtr_t)pSrc2, srcSize, maxSize, (RppPtr_t)pDst, noOfImages, rppHandle);

        printf("\nrppi_add_u8_pkd3_batchPD_gpu status is %d\n", status);
        //sleep(10);
        free(srcSize);
	return NPP_SUCCESS;
}*/

RppStatus
rppi_add_u8_pln3_batchPD_host(RppPtr_t srcPtr1,
                              RppPtr_t srcPtr2,
                              RppiSize *srcSize,
                              RppiSize maxSrcSize,
                              RppPtr_t dstPtr,
                              Rpp32u nbatchSize,
                              rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_host_roi(roiPoints, rpp::deref(rppHandle));
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    add_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr1),
                          static_cast<Rpp8u*>(srcPtr2),
                          srcSize,
                          rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                          static_cast<Rpp8u*>(dstPtr),
                          rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                          rpp::deref(rppHandle).GetBatchSize(),
                          RPPI_CHN_PLANAR,
                          3,
                          rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_add_u8_pkd3_batchPD_host(RppPtr_t srcPtr1,
                              RppPtr_t srcPtr2,
                              RppiSize *srcSize,
                              RppiSize maxSrcSize,
                              RppPtr_t dstPtr,
                              Rpp32u nbatchSize,
                              rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_host_roi(roiPoints, rpp::deref(rppHandle));
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    add_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr1),
                          static_cast<Rpp8u*>(srcPtr2),
                          srcSize,
                          rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                          static_cast<Rpp8u*>(dstPtr),
                          rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                          rpp::deref(rppHandle).GetBatchSize(),
                          RPPI_CHN_PACKED,
                          3,
                          rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

/******************** subtract ********************/

RppStatus
rppi_subtract_u8_pln1_batchPD_host(RppPtr_t srcPtr1,
                                   RppPtr_t srcPtr2,
                                   RppiSize *srcSize,
                                   RppiSize maxSrcSize,
                                   RppPtr_t dstPtr,
                                   Rpp32u nbatchSize,
                                   rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_host_roi(roiPoints, rpp::deref(rppHandle));
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    subtract_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr1),
                               static_cast<Rpp8u*>(srcPtr2),
                               srcSize,
                               rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                               static_cast<Rpp8u*>(dstPtr),
                               rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                               rpp::deref(rppHandle).GetBatchSize(),
                               RPPI_CHN_PLANAR,
                               1,
                               rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_subtract_u8_pln3_batchPD_host(RppPtr_t srcPtr1,
                                   RppPtr_t srcPtr2,
                                   RppiSize *srcSize,
                                   RppiSize maxSrcSize,
                                   RppPtr_t dstPtr,
                                   Rpp32u nbatchSize,
                                   rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_host_roi(roiPoints, rpp::deref(rppHandle));
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    subtract_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr1),
                               static_cast<Rpp8u*>(srcPtr2),
                               srcSize,
                               rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                               static_cast<Rpp8u*>(dstPtr),
                               rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                               rpp::deref(rppHandle).GetBatchSize(),
                               RPPI_CHN_PLANAR,
                               3,
                               rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_subtract_u8_pkd3_batchPD_host(RppPtr_t srcPtr1,
                                   RppPtr_t srcPtr2,
                                   RppiSize *srcSize,
                                   RppiSize maxSrcSize,
                                   RppPtr_t dstPtr,
                                   Rpp32u nbatchSize,
                                   rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_host_roi(roiPoints, rpp::deref(rppHandle));
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    subtract_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr1),
                               static_cast<Rpp8u*>(srcPtr2),
                               srcSize,
                               rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                               static_cast<Rpp8u*>(dstPtr),
                               rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                               rpp::deref(rppHandle).GetBatchSize(),
                               RPPI_CHN_PACKED,
                               3,
                               rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

/******************** multiply ********************/

RppStatus
rppi_multiply_u8_pln1_batchPD_host(RppPtr_t srcPtr1,
                                   RppPtr_t srcPtr2,
                                   RppiSize *srcSize,
                                   RppiSize maxSrcSize,
                                   RppPtr_t dstPtr,
                                   Rpp32u nbatchSize,
                                   rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_host_roi(roiPoints, rpp::deref(rppHandle));
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    multiply_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr1),
                               static_cast<Rpp8u*>(srcPtr2),
                               srcSize,
                               rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                               static_cast<Rpp8u*>(dstPtr),
                               rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                               rpp::deref(rppHandle).GetBatchSize(),
                               RPPI_CHN_PLANAR,
                               1,
                               rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_multiply_u8_pln3_batchPD_host(RppPtr_t srcPtr1,
                                   RppPtr_t srcPtr2,
                                   RppiSize *srcSize,
                                   RppiSize maxSrcSize,
                                   RppPtr_t dstPtr,
                                   Rpp32u nbatchSize,
                                   rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_host_roi(roiPoints, rpp::deref(rppHandle));
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    multiply_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr1),
                               static_cast<Rpp8u*>(srcPtr2),
                               srcSize,
                               rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                               static_cast<Rpp8u*>(dstPtr),
                               rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                               rpp::deref(rppHandle).GetBatchSize(),
                               RPPI_CHN_PLANAR,
                               3,
                               rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_multiply_u8_pkd3_batchPD_host(RppPtr_t srcPtr1,
                                   RppPtr_t srcPtr2,
                                   RppiSize *srcSize,
                                   RppiSize maxSrcSize,
                                   RppPtr_t dstPtr,
                                   Rpp32u nbatchSize,
                                   rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_host_roi(roiPoints, rpp::deref(rppHandle));
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    multiply_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr1),
                               static_cast<Rpp8u*>(srcPtr2),
                               srcSize,
                               rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                               static_cast<Rpp8u*>(dstPtr),
                               rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                               rpp::deref(rppHandle).GetBatchSize(),
                               RPPI_CHN_PACKED,
                               3,
                               rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

/******************** absolute_difference ********************/

RppStatus
rppi_absolute_difference_u8_pln1_batchPD_host(RppPtr_t srcPtr1,
                                              RppPtr_t srcPtr2,
                                              RppiSize *srcSize,
                                              RppiSize maxSrcSize,
                                              RppPtr_t dstPtr,
                                              Rpp32u nbatchSize,
                                              rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_host_roi(roiPoints, rpp::deref(rppHandle));
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    absolute_difference_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr1),
                                          static_cast<Rpp8u*>(srcPtr2),
                                          srcSize,
                                          rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                          static_cast<Rpp8u*>(dstPtr),
                                          rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                                          rpp::deref(rppHandle).GetBatchSize(),
                                          RPPI_CHN_PLANAR,
                                          1,
                                          rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_absolute_difference_u8_pln3_batchPD_host(RppPtr_t srcPtr1,
                                              RppPtr_t srcPtr2,
                                              RppiSize *srcSize,
                                              RppiSize maxSrcSize,
                                              RppPtr_t dstPtr,
                                              Rpp32u nbatchSize,
                                              rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_host_roi(roiPoints, rpp::deref(rppHandle));
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    absolute_difference_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr1),
                                          static_cast<Rpp8u*>(srcPtr2),
                                          srcSize,
                                          rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                          static_cast<Rpp8u*>(dstPtr),
                                          rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                                          rpp::deref(rppHandle).GetBatchSize(),
                                          RPPI_CHN_PLANAR,
                                          3,
                                          rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_absolute_difference_u8_pkd3_batchPD_host(RppPtr_t srcPtr1,
                                              RppPtr_t srcPtr2,
                                              RppiSize *srcSize,
                                              RppiSize maxSrcSize,
                                              RppPtr_t dstPtr,
                                              Rpp32u nbatchSize,
                                              rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_host_roi(roiPoints, rpp::deref(rppHandle));
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    absolute_difference_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr1),
                                          static_cast<Rpp8u*>(srcPtr2),
                                          srcSize,
                                          rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                          static_cast<Rpp8u*>(dstPtr),
                                          rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                                          rpp::deref(rppHandle).GetBatchSize(),
                                          RPPI_CHN_PACKED,
                                          3,
                                          rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

/******************** phase ********************/

RppStatus
rppi_phase_u8_pln1_batchPD_host(RppPtr_t srcPtr1,
                                RppPtr_t srcPtr2,
                                RppiSize *srcSize,
                                RppiSize maxSrcSize,
                                RppPtr_t dstPtr,
                                Rpp32u nbatchSize,
                                rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_host_roi(roiPoints, rpp::deref(rppHandle));
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    phase_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr1),
                            static_cast<Rpp8u*>(srcPtr2),
                            srcSize,
                            rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                            static_cast<Rpp8u*>(dstPtr),
                            rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                            rpp::deref(rppHandle).GetBatchSize(),
                            RPPI_CHN_PLANAR,
                            1,
                            rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_phase_u8_pln3_batchPD_host(RppPtr_t srcPtr1,
                                RppPtr_t srcPtr2,
                                RppiSize *srcSize,
                                RppiSize maxSrcSize,
                                RppPtr_t dstPtr,
                                Rpp32u nbatchSize,
                                rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_host_roi(roiPoints, rpp::deref(rppHandle));
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    phase_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr1),
                            static_cast<Rpp8u*>(srcPtr2),
                            srcSize,
                            rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                            static_cast<Rpp8u*>(dstPtr),
                            rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                            rpp::deref(rppHandle).GetBatchSize(),
                            RPPI_CHN_PLANAR,
                            3,
                            rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_phase_u8_pkd3_batchPD_host(RppPtr_t srcPtr1,
                                RppPtr_t srcPtr2,
                                RppiSize *srcSize,
                                RppiSize maxSrcSize,
                                RppPtr_t dstPtr,
                                Rpp32u nbatchSize,
                                rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_host_roi(roiPoints, rpp::deref(rppHandle));
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    phase_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr1),
                            static_cast<Rpp8u*>(srcPtr2),
                            srcSize,
                            rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                            static_cast<Rpp8u*>(dstPtr),
                            rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                            rpp::deref(rppHandle).GetBatchSize(),
                            RPPI_CHN_PACKED,
                            3,
                            rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

/******************** magnitude ********************/

RppStatus
rppi_magnitude_u8_pln1_batchPD_host(RppPtr_t srcPtr1,
                                    RppPtr_t srcPtr2,
                                    RppiSize *srcSize,
                                    RppiSize maxSrcSize,
                                    RppPtr_t dstPtr,
                                    Rpp32u nbatchSize,
                                    rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_host_roi(roiPoints, rpp::deref(rppHandle));
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    magnitude_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr1),
                                static_cast<Rpp8u*>(srcPtr2),
                                srcSize,
                                rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                static_cast<Rpp8u*>(dstPtr),
                                rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                                rpp::deref(rppHandle).GetBatchSize(),
                                RPPI_CHN_PLANAR,
                                1,
                                rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_magnitude_u8_pln3_batchPD_host(RppPtr_t srcPtr1,
                                    RppPtr_t srcPtr2,
                                    RppiSize *srcSize,
                                    RppiSize maxSrcSize,
                                    RppPtr_t dstPtr,
                                    Rpp32u nbatchSize,
                                    rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_host_roi(roiPoints, rpp::deref(rppHandle));
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    magnitude_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr1),
                                static_cast<Rpp8u*>(srcPtr2),
                                srcSize,
                                rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                static_cast<Rpp8u*>(dstPtr),
                                rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                                rpp::deref(rppHandle).GetBatchSize(),
                                RPPI_CHN_PLANAR,
                                3,
                                rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_magnitude_u8_pkd3_batchPD_host(RppPtr_t srcPtr1,
                                    RppPtr_t srcPtr2,
                                    RppiSize *srcSize,
                                    RppiSize maxSrcSize,
                                    RppPtr_t dstPtr,
                                    Rpp32u nbatchSize,
                                    rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_host_roi(roiPoints, rpp::deref(rppHandle));
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    magnitude_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr1),
                                static_cast<Rpp8u*>(srcPtr2),
                                srcSize,
                                rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                static_cast<Rpp8u*>(dstPtr),
                                rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                                rpp::deref(rppHandle).GetBatchSize(),
                                RPPI_CHN_PACKED,
                                3,
                                rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

/******************** accumulate ********************/

RppStatus
rppi_accumulate_u8_pln1_batchPD_host(RppPtr_t srcPtr1,
                                     RppPtr_t srcPtr2,
                                     RppiSize *srcSize,
                                     RppiSize maxSrcSize,
                                     Rpp32u nbatchSize,
                                     rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_host_roi(roiPoints, rpp::deref(rppHandle));
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    accumulate_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr1),
                                 static_cast<Rpp8u*>(srcPtr2),
                                 srcSize,
                                 rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                 rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                                 rpp::deref(rppHandle).GetBatchSize(),
                                 RPPI_CHN_PLANAR,
                                 1,
                                 rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_accumulate_u8_pln3_batchPD_host(RppPtr_t srcPtr1,
                                     RppPtr_t srcPtr2,
                                     RppiSize *srcSize,
                                     RppiSize maxSrcSize,
                                     Rpp32u nbatchSize,
                                     rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_host_roi(roiPoints, rpp::deref(rppHandle));
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    accumulate_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr1),
                                 static_cast<Rpp8u*>(srcPtr2),
                                 srcSize,
                                 rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                 rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                                 rpp::deref(rppHandle).GetBatchSize(),
                                 RPPI_CHN_PLANAR,
                                 3,
                                 rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_accumulate_u8_pkd3_batchPD_host(RppPtr_t srcPtr1,
                                     RppPtr_t srcPtr2,
                                     RppiSize *srcSize,
                                     RppiSize maxSrcSize,
                                     Rpp32u nbatchSize,
                                     rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_host_roi(roiPoints, rpp::deref(rppHandle));
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    accumulate_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr1),
                                 static_cast<Rpp8u*>(srcPtr2),
                                 srcSize,
                                 rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                 rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                                 rpp::deref(rppHandle).GetBatchSize(),
                                 RPPI_CHN_PACKED,
                                 3,
                                 rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

/******************** accumulate_weighted ********************/

RppStatus
rppi_accumulate_weighted_u8_pln1_batchPD_host(RppPtr_t srcPtr1,
                                              RppPtr_t srcPtr2,
                                              RppiSize *srcSize,
                                              RppiSize maxSrcSize,
                                              Rpp32f *alpha,
                                              Rpp32u nbatchSize,
                                              rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_host_roi(roiPoints, rpp::deref(rppHandle));
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    accumulate_weighted_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr1),
                                          static_cast<Rpp8u*>(srcPtr2),
                                          srcSize,
                                          rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                          alpha,
                                          rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                                          rpp::deref(rppHandle).GetBatchSize(),
                                          RPPI_CHN_PLANAR,
                                          1,
                                          rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_accumulate_weighted_u8_pln3_batchPD_host(RppPtr_t srcPtr1,
                                              RppPtr_t srcPtr2,
                                              RppiSize *srcSize,
                                              RppiSize maxSrcSize,
                                              Rpp32f *alpha,
                                              Rpp32u nbatchSize,
                                              rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_host_roi(roiPoints, rpp::deref(rppHandle));
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    accumulate_weighted_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr1),
                                          static_cast<Rpp8u*>(srcPtr2),
                                          srcSize,
                                          rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                          alpha,
                                          rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                                          rpp::deref(rppHandle).GetBatchSize(),
                                          RPPI_CHN_PLANAR,
                                          3,
                                          rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_accumulate_weighted_u8_pkd3_batchPD_host(RppPtr_t srcPtr1,
                                              RppPtr_t srcPtr2,
                                              RppiSize *srcSize,
                                              RppiSize maxSrcSize,
                                              Rpp32f *alpha,
                                              Rpp32u nbatchSize,
                                              rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_host_roi(roiPoints, rpp::deref(rppHandle));
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    accumulate_weighted_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr1),
                                          static_cast<Rpp8u*>(srcPtr2),
                                          srcSize,
                                          rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                          alpha,
                                          rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                                          rpp::deref(rppHandle).GetBatchSize(),
                                          RPPI_CHN_PACKED,
                                          3,
                                          rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

/******************** accumulate_squared ********************/

RppStatus
rppi_accumulate_squared_u8_pln1_batchPD_host(RppPtr_t srcPtr,
                                             RppiSize *srcSize,
                                             RppiSize maxSrcSize,
                                             Rpp32u nbatchSize,
                                             rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_host_roi(roiPoints, rpp::deref(rppHandle));
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    accumulate_squared_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                                         srcSize,
                                         rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                         rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                                         rpp::deref(rppHandle).GetBatchSize(),
                                         RPPI_CHN_PLANAR,
                                         1,
                                         rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_accumulate_squared_u8_pln3_batchPD_host(RppPtr_t srcPtr,
                                             RppiSize *srcSize,
                                             RppiSize maxSrcSize,
                                             Rpp32u nbatchSize,
                                             rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_host_roi(roiPoints, rpp::deref(rppHandle));
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    accumulate_squared_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                                         srcSize,
                                         rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                         rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                                         rpp::deref(rppHandle).GetBatchSize(),
                                         RPPI_CHN_PLANAR,
                                         3,
                                         rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_accumulate_squared_u8_pkd3_batchPD_host(RppPtr_t srcPtr,
                                             RppiSize *srcSize,
                                             RppiSize maxSrcSize,
                                             Rpp32u nbatchSize,
                                             rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_host_roi(roiPoints, rpp::deref(rppHandle));
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    accumulate_squared_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                                         srcSize,
                                         rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                         rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                                         rpp::deref(rppHandle).GetBatchSize(),
                                         RPPI_CHN_PACKED,
                                         3,
                                         rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

/******************** tensor_add ********************/

RppStatus
rppi_tensor_add_u8_host(RppPtr_t srcPtr1,
                        RppPtr_t srcPtr2,
                        RppPtr_t dstPtr,
                        Rpp32u tensorDimension,
                        RppPtr_t tensorDimensionValues,
                        rppHandle_t rppHandle)
{
    tensor_add_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr1),
                           static_cast<Rpp8u*>(srcPtr2),
                           static_cast<Rpp8u*>(dstPtr),
                           tensorDimension,
                           static_cast<Rpp32u*>(tensorDimensionValues));

    return RPP_SUCCESS;

}

/******************** tensor_subtract ********************/

RppStatus
rppi_tensor_subtract_u8_host(RppPtr_t srcPtr1,
                             RppPtr_t srcPtr2,
                             RppPtr_t dstPtr,
                             Rpp32u tensorDimension,
                             RppPtr_t tensorDimensionValues,
                             rppHandle_t rppHandle)
{
    tensor_subtract_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr1),
                                static_cast<Rpp8u*>(srcPtr2),
                                static_cast<Rpp8u*>(dstPtr),
                                tensorDimension,
                                static_cast<Rpp32u*>(tensorDimensionValues));

    return RPP_SUCCESS;

}

/******************** tensor_multiply ********************/

RppStatus
rppi_tensor_multiply_u8_host(RppPtr_t srcPtr1,
                             RppPtr_t srcPtr2,
                             RppPtr_t dstPtr,
                             Rpp32u tensorDimension,
                             RppPtr_t tensorDimensionValues,
                             rppHandle_t rppHandle)
{
    tensor_multiply_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr1),
                                static_cast<Rpp8u*>(srcPtr2),
                                static_cast<Rpp8u*>(dstPtr),
                                tensorDimension,
                                static_cast<Rpp32u*>(tensorDimensionValues));

    return RPP_SUCCESS;

}

/********************************************************************************************************************/
/*********************************************** RPP_GPU_SUPPORT = ON ***********************************************/
/********************************************************************************************************************/

#ifdef GPU_SUPPORT

/******************** add ********************/

/*RppStatus
rppi_add_16s_pkd3_batchPD_gpu(RppPtr_t srcPtr1,
                             RppPtr_t srcPtr2,
                             RppiSize *srcSize,
                             RppiSize maxSrcSize,
                             RppPtr_t dstPtr,
                             Rpp32u nbatchSize,
                             rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    copy_roi(roiPoints, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);


    add_hiptonpp_batch(static_cast<Rpp16s*>(srcPtr1),
                       static_cast<Rpp16s*>(srcPtr2),
                       static_cast<Rpp16s*>(dstPtr),
                       rpp::deref(rppHandle),
                       RPPI_CHN_PACKED,
                       3);

    return RPP_SUCCESS;
}

RppStatus
rppi_add_16f_pkd3_batchPD_gpu(RppPtr_t srcPtr1,
                             RppPtr_t srcPtr2,
                             RppiSize *srcSize,
                             RppiSize maxSrcSize,
                             RppPtr_t dstPtr,
                             Rpp32u nbatchSize,
                             rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    copy_roi(roiPoints, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);


    add_hiptonpp_batch(reinterpret_cast<half*>(static_cast<Rpp8u*>(srcPtr1)),
                       reinterpret_cast<half*>(static_cast<Rpp8u*>(srcPtr2)),
                       reinterpret_cast<half*>(static_cast<Rpp8u*>(dstPtr)),
                       rpp::deref(rppHandle),
                       RPPI_CHN_PACKED,
                       3);

    return RPP_SUCCESS;
}*/

RppStatus
rppi_add_32f_pkd3_batchPD_gpu(RppPtr_t srcPtr1,
                             RppPtr_t srcPtr2,
                             RppiSize *srcSize,
                             RppiSize maxSrcSize,
                             RppPtr_t dstPtr,
                             Rpp32u nbatchSize,
                             rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    copy_roi(roiPoints, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);


    add_hip32f_batch(static_cast<Rpp32f*>(srcPtr1),
                     static_cast<Rpp32f*>(srcPtr2),
                     static_cast<Rpp32f*>(dstPtr),
                     rpp::deref(rppHandle),
                     RPPI_CHN_PACKED,
                     3);

    return RPP_SUCCESS;
}

RppStatus
rppi_add_u8_p4_batchPD_gpu(RppPtr_t srcPtr1,
                             RppPtr_t srcPtr2,
                             RppiSize *srcSize,
                             RppiSize maxSrcSize,
                             RppPtr_t dstPtr,
                             Rpp32u nbatchSize,
                             rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    copy_roi(roiPoints, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), 4, RPPI_CHN_PLANAR);


    add_hip_batch(static_cast<Rpp8u*>(srcPtr1),
                     static_cast<Rpp8u*>(srcPtr2),
                     static_cast<Rpp8u*>(dstPtr),
                     rpp::deref(rppHandle),
                     RPPI_CHN_PLANAR,
                     4);

    return RPP_SUCCESS;
}

RppStatus
rppi_add_u8_pln1_batchPD_gpu(RppPtr_t srcPtr1,
                             RppPtr_t srcPtr2,
                             RppiSize *srcSize,
                             RppiSize maxSrcSize,
                             RppPtr_t dstPtr,
                             Rpp32u nbatchSize,
                             rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    copy_roi(roiPoints, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
    {
        add_cl_batch(static_cast<cl_mem>(srcPtr1),
                     static_cast<cl_mem>(srcPtr2),
                     static_cast<cl_mem>(dstPtr),
                     rpp::deref(rppHandle),
                     RPPI_CHN_PLANAR,
                     1);
    }
#elif defined(HIP_COMPILE)
    {
        add_hip_batch(static_cast<Rpp8u*>(srcPtr1),
                      static_cast<Rpp8u*>(srcPtr2),
                      static_cast<Rpp8u*>(dstPtr),
                      rpp::deref(rppHandle),
                      RPPI_CHN_PLANAR,
                      1);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_add_u8_pln3_batchPD_gpu(RppPtr_t srcPtr1,
                             RppPtr_t srcPtr2,
                             RppiSize *srcSize,
                             RppiSize maxSrcSize,
                             RppPtr_t dstPtr,
                             Rpp32u nbatchSize,
                             rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    copy_roi(roiPoints, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
    {
        add_cl_batch(static_cast<cl_mem>(srcPtr1),
                     static_cast<cl_mem>(srcPtr2),
                     static_cast<cl_mem>(dstPtr),
                     rpp::deref(rppHandle),
                     RPPI_CHN_PLANAR,
                     3);
    }
#elif defined(HIP_COMPILE)
    {
        add_hip_batch(static_cast<Rpp8u*>(srcPtr1),
                      static_cast<Rpp8u*>(srcPtr2),
                      static_cast<Rpp8u*>(dstPtr),
                      rpp::deref(rppHandle),
                      RPPI_CHN_PLANAR,
                      3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_add_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr1,
                             RppPtr_t srcPtr2,
                             RppiSize *srcSize,
                             RppiSize maxSrcSize,
                             RppPtr_t dstPtr,
                             Rpp32u nbatchSize,
                             rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    copy_roi(roiPoints, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
    {
        add_cl_batch(static_cast<cl_mem>(srcPtr1),
                     static_cast<cl_mem>(srcPtr2),
                     static_cast<cl_mem>(dstPtr),
                     rpp::deref(rppHandle),
                     RPPI_CHN_PACKED,
                     3);
    }
#elif defined(HIP_COMPILE)
    {
        add_hip_batch(static_cast<Rpp8u*>(srcPtr1),
                      static_cast<Rpp8u*>(srcPtr2),
                      static_cast<Rpp8u*>(dstPtr),
                      rpp::deref(rppHandle),
                      RPPI_CHN_PACKED,
                      3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

/******************** subtract ********************/

RppStatus
rppi_subtract_u8_pln1_batchPD_gpu(RppPtr_t srcPtr1,
                                  RppPtr_t srcPtr2,
                                  RppiSize *srcSize,
                                  RppiSize maxSrcSize,
                                  RppPtr_t dstPtr,
                                  Rpp32u nbatchSize,
                                  rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    copy_roi(roiPoints, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
    {
        subtract_cl_batch(static_cast<cl_mem>(srcPtr1),
                          static_cast<cl_mem>(srcPtr2),
                          static_cast<cl_mem>(dstPtr),
                          rpp::deref(rppHandle),
                          RPPI_CHN_PLANAR,
                          1);
    }
#elif defined(HIP_COMPILE)
    {
        subtract_hip_batch(static_cast<Rpp8u*>(srcPtr1),
                           static_cast<Rpp8u*>(srcPtr2),
                           static_cast<Rpp8u*>(dstPtr),
                           rpp::deref(rppHandle),
                           RPPI_CHN_PLANAR,
                           1);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_subtract_u8_pln3_batchPD_gpu(RppPtr_t srcPtr1,
                                  RppPtr_t srcPtr2,
                                  RppiSize *srcSize,
                                  RppiSize maxSrcSize,
                                  RppPtr_t dstPtr,
                                  Rpp32u nbatchSize,
                                  rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    copy_roi(roiPoints, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
    {
        subtract_cl_batch(static_cast<cl_mem>(srcPtr1),
                          static_cast<cl_mem>(srcPtr2),
                          static_cast<cl_mem>(dstPtr),
                          rpp::deref(rppHandle),
                          RPPI_CHN_PLANAR,
                          3);
    }
#elif defined(HIP_COMPILE)
    {
        subtract_hip_batch(static_cast<Rpp8u*>(srcPtr1),
                           static_cast<Rpp8u*>(srcPtr2),
                           static_cast<Rpp8u*>(dstPtr),
                           rpp::deref(rppHandle),
                           RPPI_CHN_PLANAR,
                           3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_subtract_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr1,
                                  RppPtr_t srcPtr2,
                                  RppiSize *srcSize,
                                  RppiSize maxSrcSize,
                                  RppPtr_t dstPtr,
                                  Rpp32u nbatchSize,
                                  rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    copy_roi(roiPoints, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
    {
        subtract_cl_batch(static_cast<cl_mem>(srcPtr1),
                          static_cast<cl_mem>(srcPtr2),
                          static_cast<cl_mem>(dstPtr),
                          rpp::deref(rppHandle),
                          RPPI_CHN_PACKED,
                          3);
    }
#elif defined(HIP_COMPILE)
    {
        subtract_hip_batch(static_cast<Rpp8u*>(srcPtr1),
                           static_cast<Rpp8u*>(srcPtr2),
                           static_cast<Rpp8u*>(dstPtr),
                           rpp::deref(rppHandle),
                           RPPI_CHN_PACKED,
                           3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

/******************** multiply ********************/

RppStatus
rppi_multiply_u8_pln1_batchPD_gpu(RppPtr_t srcPtr1,
                                  RppPtr_t srcPtr2,
                                  RppiSize *srcSize,
                                  RppiSize maxSrcSize,
                                  RppPtr_t dstPtr,
                                  Rpp32u nbatchSize,
                                  rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    copy_roi(roiPoints, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
    {
        multiply_cl_batch(static_cast<cl_mem>(srcPtr1),
                          static_cast<cl_mem>(srcPtr2),
                          static_cast<cl_mem>(dstPtr),
                          rpp::deref(rppHandle),
                          RPPI_CHN_PLANAR,
                          1);
    }
#elif defined(HIP_COMPILE)
    {
        multiply_hip_batch(static_cast<Rpp8u*>(srcPtr1),
                           static_cast<Rpp8u*>(srcPtr2),
                           static_cast<Rpp8u*>(dstPtr),
                           rpp::deref(rppHandle),
                           RPPI_CHN_PLANAR,
                           1);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_multiply_u8_pln3_batchPD_gpu(RppPtr_t srcPtr1,
                                  RppPtr_t srcPtr2,
                                  RppiSize *srcSize,
                                  RppiSize maxSrcSize,
                                  RppPtr_t dstPtr,
                                  Rpp32u nbatchSize,
                                  rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    copy_roi(roiPoints, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
    {
        multiply_cl_batch(static_cast<cl_mem>(srcPtr1),
                          static_cast<cl_mem>(srcPtr2),
                          static_cast<cl_mem>(dstPtr),
                          rpp::deref(rppHandle),
                          RPPI_CHN_PLANAR,
                          3);
    }
#elif defined(HIP_COMPILE)
    {
        multiply_hip_batch(static_cast<Rpp8u*>(srcPtr1),
                           static_cast<Rpp8u*>(srcPtr2),
                           static_cast<Rpp8u*>(dstPtr),
                           rpp::deref(rppHandle),
                           RPPI_CHN_PLANAR,
                           3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_multiply_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr1,
                                  RppPtr_t srcPtr2,
                                  RppiSize *srcSize,
                                  RppiSize maxSrcSize,
                                  RppPtr_t dstPtr,
                                  Rpp32u nbatchSize,
                                  rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    copy_roi(roiPoints, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
    {
        multiply_cl_batch(static_cast<cl_mem>(srcPtr1),
                          static_cast<cl_mem>(srcPtr2),
                          static_cast<cl_mem>(dstPtr),
                          rpp::deref(rppHandle),
                          RPPI_CHN_PACKED,
                          3);
    }
#elif defined(HIP_COMPILE)
    {
        multiply_hip_batch(static_cast<Rpp8u*>(srcPtr1),
                           static_cast<Rpp8u*>(srcPtr2),
                           static_cast<Rpp8u*>(dstPtr),
                           rpp::deref(rppHandle),
                           RPPI_CHN_PACKED,
                           3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_div_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr1,
                                  RppPtr_t srcPtr2,
                                  RppiSize *srcSize,
                                  RppiSize maxSrcSize,
                                  RppPtr_t dstPtr,
                                  Rpp32u nbatchSize,
                                  rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    copy_roi(roiPoints, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

    div_hip_batch(static_cast<Rpp8u*>(srcPtr1),
                       static_cast<Rpp8u*>(srcPtr2),
                       static_cast<Rpp8u*>(dstPtr),
                       rpp::deref(rppHandle),
                       RPPI_CHN_PACKED,
                       3);

    return RPP_SUCCESS;
}

NppStatus nppiDiv_8u_C3RSfs(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor)
{
        int noOfImages = 1;
        int ip_channel = 3;
        //unsigned long long oBufferSize = 0;
        RppiSize *srcSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize));
        RppiSize maxSize;
        srcSize->width  = oSizeROI.width;
        srcSize->height = oSizeROI.height;
        maxSize.width  = oSizeROI.width;
        maxSize.height = oSizeROI.height;
        //oBufferSize = (unsigned long long)maxSize.height * (unsigned long long)maxSize.width * (unsigned long long)ip_channel * (unsigned long long)noOfImages;

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        status = rppi_div_u8_pkd3_batchPD_gpu((RppPtr_t)pSrc1, (RppPtr_t)pSrc2, srcSize, maxSize, (RppPtr_t)pDst, noOfImages, handle);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        free(srcSize);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiAdd_8u_AC4RSfs(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor)
{
        int noOfImages = 1;
        int ip_channel = 4;
        //unsigned long long oBufferSize = 0;
        RppiSize *srcSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize));
        RppiSize maxSize;
        srcSize->width  = oSizeROI.width;
        srcSize->height = oSizeROI.height;
        maxSize.width  = oSizeROI.width;
        maxSize.height = oSizeROI.height;
        //oBufferSize = (unsigned long long)maxSize.height * (unsigned long long)maxSize.width * (unsigned long long)ip_channel * (unsigned long long)noOfImages;

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        status = rppi_add_u8_p4_batchPD_gpu((RppPtr_t)pSrc1, (RppPtr_t)pSrc2, srcSize, maxSize, (RppPtr_t)pDst, noOfImages, handle);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        free(srcSize);

        return(hipRppStatusTocudaNppStatus(status));
}

/******************** absolute_difference ********************/

RppStatus
rppi_absolute_difference_u8_pln1_batchPD_gpu(RppPtr_t srcPtr1,
                                             RppPtr_t srcPtr2,
                                             RppiSize *srcSize,
                                             RppiSize maxSrcSize,
                                             RppPtr_t dstPtr,
                                             Rpp32u nbatchSize,
                                             rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    copy_roi(roiPoints, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
    {
        absolute_difference_cl_batch(static_cast<cl_mem>(srcPtr1),
                                     static_cast<cl_mem>(srcPtr2),
                                     static_cast<cl_mem>(dstPtr),
                                     rpp::deref(rppHandle),
                                     RPPI_CHN_PLANAR,
                                     1);
    }
#elif defined(HIP_COMPILE)
    {
        absolute_difference_hip_batch(static_cast<Rpp8u*>(srcPtr1),
                                      static_cast<Rpp8u*>(srcPtr2),
                                      static_cast<Rpp8u*>(dstPtr),
                                      rpp::deref(rppHandle),
                                      RPPI_CHN_PLANAR,
                                      1);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_absolute_difference_u8_pln3_batchPD_gpu(RppPtr_t srcPtr1,
                                             RppPtr_t srcPtr2,
                                             RppiSize *srcSize,
                                             RppiSize maxSrcSize,
                                             RppPtr_t dstPtr,
                                             Rpp32u nbatchSize,
                                             rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    copy_roi(roiPoints, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
    {
        absolute_difference_cl_batch(static_cast<cl_mem>(srcPtr1),
                                     static_cast<cl_mem>(srcPtr2),
                                     static_cast<cl_mem>(dstPtr),
                                     rpp::deref(rppHandle),
                                     RPPI_CHN_PLANAR,
                                     3);
    }
#elif defined(HIP_COMPILE)
    {
        absolute_difference_hip_batch(static_cast<Rpp8u*>(srcPtr1),
                                      static_cast<Rpp8u*>(srcPtr2),
                                      static_cast<Rpp8u*>(dstPtr),
                                      rpp::deref(rppHandle),
                                      RPPI_CHN_PLANAR,
                                      3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_absolute_difference_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr1,
                                             RppPtr_t srcPtr2,
                                             RppiSize *srcSize,
                                             RppiSize maxSrcSize,
                                             RppPtr_t dstPtr,
                                             Rpp32u nbatchSize,
                                             rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    copy_roi(roiPoints, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
    {
        absolute_difference_cl_batch(static_cast<cl_mem>(srcPtr1),
                                     static_cast<cl_mem>(srcPtr2),
                                     static_cast<cl_mem>(dstPtr),
                                     rpp::deref(rppHandle),
                                     RPPI_CHN_PACKED,
                                     3);
    }
#elif defined(HIP_COMPILE)
    {
        absolute_difference_hip_batch(static_cast<Rpp8u*>(srcPtr1),
                                      static_cast<Rpp8u*>(srcPtr2),
                                      static_cast<Rpp8u*>(dstPtr),
                                      rpp::deref(rppHandle),
                                      RPPI_CHN_PACKED,
                                      3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

/******************** phase ********************/

RppStatus
rppi_phase_u8_pln1_batchPD_gpu(RppPtr_t srcPtr1,
                               RppPtr_t srcPtr2,
                               RppiSize *srcSize,
                               RppiSize maxSrcSize,
                               RppPtr_t dstPtr,
                               Rpp32u nbatchSize,
                               rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    copy_roi(roiPoints, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
    {
        phase_cl_batch(static_cast<cl_mem>(srcPtr1),
                       static_cast<cl_mem>(srcPtr2),
                       static_cast<cl_mem>(dstPtr),
                       rpp::deref(rppHandle),
                       RPPI_CHN_PLANAR,
                       1);
    }
#elif defined(HIP_COMPILE)
    {
        phase_hip_batch(static_cast<Rpp8u*>(srcPtr1),
                        static_cast<Rpp8u*>(srcPtr2),
                        static_cast<Rpp8u*>(dstPtr),
                        rpp::deref(rppHandle),
                        RPPI_CHN_PLANAR,
                        1);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_phase_u8_pln3_batchPD_gpu(RppPtr_t srcPtr1,
                               RppPtr_t srcPtr2,
                               RppiSize *srcSize,
                               RppiSize maxSrcSize,
                               RppPtr_t dstPtr,
                               Rpp32u nbatchSize,
                               rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    copy_roi(roiPoints, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
    {
        phase_cl_batch(static_cast<cl_mem>(srcPtr1),
                       static_cast<cl_mem>(srcPtr2),
                       static_cast<cl_mem>(dstPtr),
                       rpp::deref(rppHandle),
                       RPPI_CHN_PLANAR,
                       3);
    }
#elif defined(HIP_COMPILE)
    {
        phase_hip_batch(static_cast<Rpp8u*>(srcPtr1),
                        static_cast<Rpp8u*>(srcPtr2),
                        static_cast<Rpp8u*>(dstPtr),
                        rpp::deref(rppHandle),
                        RPPI_CHN_PLANAR,
                        3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_phase_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr1,
                               RppPtr_t srcPtr2,
                               RppiSize *srcSize,
                               RppiSize maxSrcSize,
                               RppPtr_t dstPtr,
                               Rpp32u nbatchSize,
                               rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    copy_roi(roiPoints, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
    {
        phase_cl_batch(static_cast<cl_mem>(srcPtr1),
                       static_cast<cl_mem>(srcPtr2),
                       static_cast<cl_mem>(dstPtr),
                       rpp::deref(rppHandle),
                       RPPI_CHN_PACKED,
                       3);
    }
#elif defined(HIP_COMPILE)
    {
        phase_hip_batch(static_cast<Rpp8u*>(srcPtr1),
                        static_cast<Rpp8u*>(srcPtr2),
                        static_cast<Rpp8u*>(dstPtr),
                        rpp::deref(rppHandle),
                        RPPI_CHN_PACKED,
                        3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

/******************** magnitude ********************/

RppStatus
rppi_magnitude_u8_pln1_batchPD_gpu(RppPtr_t srcPtr1,
                                   RppPtr_t srcPtr2,
                                   RppiSize *srcSize,
                                   RppiSize maxSrcSize,
                                   RppPtr_t dstPtr,
                                   Rpp32u nbatchSize,
                                   rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    copy_roi(roiPoints, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
    {
        magnitude_cl_batch(static_cast<cl_mem>(srcPtr1),
                           static_cast<cl_mem>(srcPtr2),
                           static_cast<cl_mem>(dstPtr),
                           rpp::deref(rppHandle),
                           RPPI_CHN_PLANAR,
                           1);
    }
#elif defined(HIP_COMPILE)
    {
        magnitude_hip_batch(static_cast<Rpp8u*>(srcPtr1),
                            static_cast<Rpp8u*>(srcPtr2),
                            static_cast<Rpp8u*>(dstPtr),
                            rpp::deref(rppHandle),
                            RPPI_CHN_PLANAR,
                            1);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_magnitude_u8_pln3_batchPD_gpu(RppPtr_t srcPtr1,
                                   RppPtr_t srcPtr2,
                                   RppiSize *srcSize,
                                   RppiSize maxSrcSize,
                                   RppPtr_t dstPtr,
                                   Rpp32u nbatchSize,
                                   rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    copy_roi(roiPoints, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
    {
        magnitude_cl_batch(static_cast<cl_mem>(srcPtr1),
                           static_cast<cl_mem>(srcPtr2),
                           static_cast<cl_mem>(dstPtr),
                           rpp::deref(rppHandle),
                           RPPI_CHN_PLANAR,
                           3);
    }
#elif defined(HIP_COMPILE)
    {
        magnitude_hip_batch(static_cast<Rpp8u*>(srcPtr1),
                            static_cast<Rpp8u*>(srcPtr2),
                            static_cast<Rpp8u*>(dstPtr),
                            rpp::deref(rppHandle),
                            RPPI_CHN_PLANAR,
                            3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_magnitude_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr1,
                                   RppPtr_t srcPtr2,
                                   RppiSize *srcSize,
                                   RppiSize maxSrcSize,
                                   RppPtr_t dstPtr,
                                   Rpp32u nbatchSize,
                                   rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    copy_roi(roiPoints, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
    {
        magnitude_cl_batch(static_cast<cl_mem>(srcPtr1),
                           static_cast<cl_mem>(srcPtr2),
                           static_cast<cl_mem>(dstPtr),
                           rpp::deref(rppHandle),
                           RPPI_CHN_PACKED,
                           3);
    }
#elif defined(HIP_COMPILE)
    {
        magnitude_hip_batch(static_cast<Rpp8u*>(srcPtr1),
                            static_cast<Rpp8u*>(srcPtr2),
                            static_cast<Rpp8u*>(dstPtr),
                            rpp::deref(rppHandle),
                            RPPI_CHN_PACKED,
                            3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

/******************** accumulate ********************/

RppStatus
rppi_accumulate_u8_pln1_batchPD_gpu(RppPtr_t srcPtr1,
                                    RppPtr_t srcPtr2,
                                    RppiSize *srcSize,
                                    RppiSize maxSrcSize,
                                    Rpp32u nbatchSize,
                                    rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    copy_roi(roiPoints, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
    {
        accumulate_cl_batch(static_cast<cl_mem>(srcPtr1),
                            static_cast<cl_mem>(srcPtr2),
                            rpp::deref(rppHandle),
                            RPPI_CHN_PLANAR,
                            1);
    }
#elif defined(HIP_COMPILE)
    {
        accumulate_hip_batch(static_cast<Rpp8u*>(srcPtr1),
                             static_cast<Rpp8u*>(srcPtr2),
                             rpp::deref(rppHandle),
                             RPPI_CHN_PLANAR,
                             1);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_accumulate_u8_pln3_batchPD_gpu(RppPtr_t srcPtr1,
                                    RppPtr_t srcPtr2,
                                    RppiSize *srcSize,
                                    RppiSize maxSrcSize,
                                    Rpp32u nbatchSize,
                                    rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    copy_roi(roiPoints, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
    {
        accumulate_cl_batch(static_cast<cl_mem>(srcPtr1),
                            static_cast<cl_mem>(srcPtr2),
                            rpp::deref(rppHandle),
                            RPPI_CHN_PLANAR,
                            3);
    }
#elif defined(HIP_COMPILE)
    {
        accumulate_hip_batch(static_cast<Rpp8u*>(srcPtr1),
                             static_cast<Rpp8u*>(srcPtr2),
                             rpp::deref(rppHandle),
                             RPPI_CHN_PLANAR,
                             3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_accumulate_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr1,
                                    RppPtr_t srcPtr2,
                                    RppiSize *srcSize,
                                    RppiSize maxSrcSize,
                                    Rpp32u nbatchSize,
                                    rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    copy_roi(roiPoints, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
    {
        accumulate_cl_batch(static_cast<cl_mem>(srcPtr1),
                            static_cast<cl_mem>(srcPtr2),
                            rpp::deref(rppHandle),
                            RPPI_CHN_PACKED,
                            3);
    }
#elif defined(HIP_COMPILE)
    {
        accumulate_hip_batch(static_cast<Rpp8u*>(srcPtr1),
                             static_cast<Rpp8u*>(srcPtr2),
                             rpp::deref(rppHandle),
                             RPPI_CHN_PACKED,
                             3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

/******************** accumulate_weighted ********************/

RppStatus
rppi_accumulate_weighted_u8_pln1_batchPD_gpu(RppPtr_t srcPtr1,
                                             RppPtr_t srcPtr2,
                                             RppiSize *srcSize,
                                             RppiSize maxSrcSize,
                                             Rpp32f *alpha,
                                             Rpp32u nbatchSize,
                                             rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    Rpp32u paramIndex = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    copy_roi(roiPoints, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
    copy_param_float(alpha, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        accumulate_weighted_cl_batch(static_cast<cl_mem>(srcPtr1),
                                     static_cast<cl_mem>(srcPtr2),
                                     rpp::deref(rppHandle),
                                     RPPI_CHN_PLANAR,
                                     1);
    }
#elif defined(HIP_COMPILE)
    {
        accumulate_weighted_hip_batch(static_cast<Rpp8u*>(srcPtr1),
                                      static_cast<Rpp8u*>(srcPtr2),
                                      rpp::deref(rppHandle),
                                      RPPI_CHN_PLANAR,
                                      1);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_accumulate_weighted_u8_pln3_batchPD_gpu(RppPtr_t srcPtr1,
                                             RppPtr_t srcPtr2,
                                             RppiSize *srcSize,
                                             RppiSize maxSrcSize,
                                             Rpp32f *alpha,
                                             Rpp32u nbatchSize,
                                             rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    Rpp32u paramIndex = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    copy_roi(roiPoints, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
    copy_param_float(alpha, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        accumulate_weighted_cl_batch(static_cast<cl_mem>(srcPtr1),
                                     static_cast<cl_mem>(srcPtr2),
                                     rpp::deref(rppHandle),
                                     RPPI_CHN_PLANAR,
                                     3);
    }
#elif defined(HIP_COMPILE)
    {
        accumulate_weighted_hip_batch(static_cast<Rpp8u*>(srcPtr1),
                                      static_cast<Rpp8u*>(srcPtr2),
                                      rpp::deref(rppHandle),
                                      RPPI_CHN_PLANAR,
                                      3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_accumulate_weighted_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr1,
                                             RppPtr_t srcPtr2,
                                             RppiSize *srcSize,
                                             RppiSize maxSrcSize,
                                             Rpp32f *alpha,
                                             Rpp32u nbatchSize,
                                             rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    Rpp32u paramIndex = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    copy_roi(roiPoints, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
    copy_param_float(alpha, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        accumulate_weighted_cl_batch(static_cast<cl_mem>(srcPtr1),
                                     static_cast<cl_mem>(srcPtr2),
                                     rpp::deref(rppHandle),
                                     RPPI_CHN_PACKED,
                                     3);
    }
#elif defined(HIP_COMPILE)
    {
        accumulate_weighted_hip_batch(static_cast<Rpp8u*>(srcPtr1),
                                      static_cast<Rpp8u*>(srcPtr2),
                                      rpp::deref(rppHandle),
                                      RPPI_CHN_PACKED,
                                      3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

/******************** accumulate_squared ********************/

RppStatus
rppi_accumulate_squared_u8_pln1_batchPD_gpu(RppPtr_t srcPtr,
                                            RppiSize *srcSize,
                                            RppiSize maxSrcSize,
                                            Rpp32u nbatchSize,
                                            rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    copy_roi(roiPoints, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
    {
        accumulate_squared_cl_batch(static_cast<cl_mem>(srcPtr),
                                    rpp::deref(rppHandle),
                                    RPPI_CHN_PLANAR,
                                    1);
    }
#elif defined(HIP_COMPILE)
    {
        accumulate_squared_hip_batch(static_cast<Rpp8u*>(srcPtr),
                                     rpp::deref(rppHandle),
                                     RPPI_CHN_PLANAR,
                                     1);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_accumulate_squared_u8_pln3_batchPD_gpu(RppPtr_t srcPtr,
                                            RppiSize *srcSize,
                                            RppiSize maxSrcSize,
                                            Rpp32u nbatchSize,
                                            rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    copy_roi(roiPoints, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
    {
        accumulate_squared_cl_batch(static_cast<cl_mem>(srcPtr),
                                    rpp::deref(rppHandle),
                                    RPPI_CHN_PLANAR,
                                    3);
    }
#elif defined(HIP_COMPILE)
    {
        accumulate_squared_hip_batch(static_cast<Rpp8u*>(srcPtr),
                                     rpp::deref(rppHandle),
                                     RPPI_CHN_PLANAR,
                                     3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_accumulate_squared_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr,
                                            RppiSize *srcSize,
                                            RppiSize maxSrcSize,
                                            Rpp32u nbatchSize,
                                            rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    copy_roi(roiPoints, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
    {
        accumulate_squared_cl_batch(static_cast<cl_mem>(srcPtr),
                                    rpp::deref(rppHandle),
                                    RPPI_CHN_PACKED,
                                    3);
    }
#elif defined(HIP_COMPILE)
    {
        accumulate_squared_hip_batch(static_cast<Rpp8u*>(srcPtr),
                                     rpp::deref(rppHandle),
                                     RPPI_CHN_PACKED,
                                     3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

/******************** tensor_add ********************/

RppStatus
rppi_tensor_add_u8_gpu(RppPtr_t srcPtr1,
                       RppPtr_t srcPtr2,
                       RppPtr_t dstPtr,
                       Rpp32u tensorDimension,
                       RppPtr_t tensorDimensionValues,
                       rppHandle_t rppHandle)
{

#ifdef OCL_COMPILE
    {
        tensor_add_cl(tensorDimension,
                      static_cast<Rpp32u*>(tensorDimensionValues),
                      static_cast<cl_mem>(srcPtr1),
                      static_cast<cl_mem>(srcPtr2),
                      static_cast<cl_mem>(dstPtr),
                      rpp::deref(rppHandle));
    }
#elif defined(HIP_COMPILE)
    {
        tensor_add_hip(tensorDimension,
                       static_cast<Rpp32u*>(tensorDimensionValues),
                       static_cast<Rpp8u*>(srcPtr1),
                       static_cast<Rpp8u*>(srcPtr2),
                       static_cast<Rpp8u*>(dstPtr),
                       rpp::deref(rppHandle));
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

/******************** tensor_subtract ********************/

RppStatus
rppi_tensor_subtract_u8_gpu(RppPtr_t srcPtr1,
                            RppPtr_t srcPtr2,
                            RppPtr_t dstPtr,
                            Rpp32u tensorDimension,
                            RppPtr_t tensorDimensionValues,
                            rppHandle_t rppHandle)
{

#ifdef OCL_COMPILE
    {
        tensor_subtract_cl(tensorDimension,
                           static_cast<Rpp32u*>(tensorDimensionValues),
                           static_cast<cl_mem>(srcPtr1),
                           static_cast<cl_mem>(srcPtr2),
                           static_cast<cl_mem>(dstPtr),
                           rpp::deref(rppHandle));
    }
#elif defined(HIP_COMPILE)
    {
        tensor_subtract_hip(tensorDimension,
                            static_cast<Rpp32u*>(tensorDimensionValues),
                            static_cast<Rpp8u*>(srcPtr1),
                            static_cast<Rpp8u*>(srcPtr2),
                            static_cast<Rpp8u*>(dstPtr),
                            rpp::deref(rppHandle));
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

/******************** tensor_multiply ********************/

RppStatus
rppi_tensor_multiply_u8_gpu(RppPtr_t srcPtr1,
                            RppPtr_t srcPtr2,
                            RppPtr_t dstPtr,
                            Rpp32u tensorDimension,
                            RppPtr_t tensorDimensionValues,
                            rppHandle_t rppHandle)
{

#ifdef OCL_COMPILE
    {
        tensor_multiply_cl(tensorDimension,
                           static_cast<Rpp32u*>(tensorDimensionValues),
                           static_cast<cl_mem>(srcPtr1),
                           static_cast<cl_mem>(srcPtr2),
                           static_cast<cl_mem>(dstPtr),
                           rpp::deref(rppHandle));
    }
#elif defined(HIP_COMPILE)
    {
        tensor_multiply_hip(tensorDimension,
                            static_cast<Rpp32u*>(tensorDimensionValues),
                            static_cast<Rpp8u*>(srcPtr1),
                            static_cast<Rpp8u*>(srcPtr2),
                            static_cast<Rpp8u*>(dstPtr),
                            rpp::deref(rppHandle));
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

#endif // GPU_SUPPORT
enum cudaError hipErrorTocudaError(hipError_t para) {
    switch (para) {
//         case hipErrorAddressOfConstant:
//             return cudaErrorAddressOfConstant;
        case hipErrorAlreadyAcquired:
            return cudaErrorAlreadyAcquired;
        case hipErrorAlreadyMapped:
            return cudaErrorAlreadyMapped;
//         case hipErrorApiFailureBase:
//             return cudaErrorApiFailureBase;
        case hipErrorArrayIsMapped:
            return cudaErrorArrayIsMapped;
        case hipErrorAssert:
            return cudaErrorAssert;
        case hipErrorCapturedEvent:
            return cudaErrorCapturedEvent;
//         case hipErrorCompatNotSupportedOnDevice:
//             return cudaErrorCompatNotSupportedOnDevice;
        case hipErrorContextIsDestroyed:
            return cudaErrorContextIsDestroyed;
        case hipErrorCooperativeLaunchTooLarge:
            return cudaErrorCooperativeLaunchTooLarge;
        case hipErrorDeinitialized:
            return cudaErrorCudartUnloading;
        case hipErrorContextAlreadyInUse:
            return cudaErrorDeviceAlreadyInUse;
        case hipErrorInvalidContext:
            return cudaErrorDeviceUninitialized;
//         case hipErrorDevicesUnavailable:
//             return cudaErrorDevicesUnavailable;
//         case hipErrorDuplicateSurfaceName:
//             return cudaErrorDuplicateSurfaceName;
//         case hipErrorDuplicateTextureName:
//             return cudaErrorDuplicateTextureName;
//         case hipErrorDuplicateVariableName:
//             return cudaErrorDuplicateVariableName;
        case hipErrorECCNotCorrectable:
            return cudaErrorECCUncorrectable;
        case hipErrorFileNotFound:
            return cudaErrorFileNotFound;
        case hipErrorGraphExecUpdateFailure:
            return cudaErrorGraphExecUpdateFailure;
//         case hipErrorHardwareStackError:
//             return cudaErrorHardwareStackError;
        case hipErrorHostMemoryAlreadyRegistered:
            return cudaErrorHostMemoryAlreadyRegistered;
        case hipErrorHostMemoryNotRegistered:
            return cudaErrorHostMemoryNotRegistered;
        case hipErrorIllegalAddress:
            return cudaErrorIllegalAddress;
//         case hipErrorIllegalInstruction:
//             return cudaErrorIllegalInstruction;
        case hipErrorIllegalState:
            return cudaErrorIllegalState;
//         case hipErrorIncompatibleDriverContext:
//             return cudaErrorIncompatibleDriverContext;
        case hipErrorNotInitialized:
            return cudaErrorInitializationError;
        case hipErrorInsufficientDriver:
            return cudaErrorInsufficientDriver;
//         case hipErrorInvalidAddressSpace:
//             return cudaErrorInvalidAddressSpace;
//         case hipErrorInvalidChannelDescriptor:
//             return cudaErrorInvalidChannelDescriptor;
        case hipErrorInvalidConfiguration:
            return cudaErrorInvalidConfiguration;
        case hipErrorInvalidDevice:
            return cudaErrorInvalidDevice;
        case hipErrorInvalidDeviceFunction:
            return cudaErrorInvalidDeviceFunction;
        case hipErrorInvalidDevicePointer:
            return cudaErrorInvalidDevicePointer;
//         case hipErrorInvalidFilterSetting:
//             return cudaErrorInvalidFilterSetting;
        case hipErrorInvalidGraphicsContext:
            return cudaErrorInvalidGraphicsContext;
//         case hipErrorInvalidHostPointer:
//             return cudaErrorInvalidHostPointer;
        case hipErrorInvalidImage:
            return cudaErrorInvalidKernelImage;
        case hipErrorInvalidMemcpyDirection:
            return cudaErrorInvalidMemcpyDirection;
//         case hipErrorInvalidNormSetting:
//             return cudaErrorInvalidNormSetting;
//         case hipErrorInvalidPc:
//             return cudaErrorInvalidPc;
        case hipErrorInvalidPitchValue:
            return cudaErrorInvalidPitchValue;
        case hipErrorInvalidKernelFile:
            return cudaErrorInvalidPtx;
        case hipErrorInvalidHandle:
            return cudaErrorInvalidResourceHandle;
        case hipErrorInvalidSource:
            return cudaErrorInvalidSource;
//         case hipErrorInvalidSurface:
//             return cudaErrorInvalidSurface;
        case hipErrorInvalidSymbol:
            return cudaErrorInvalidSymbol;
//         case hipErrorInvalidTexture:
//             return cudaErrorInvalidTexture;
//         case hipErrorInvalidTextureBinding:
//             return cudaErrorInvalidTextureBinding;
        case hipErrorInvalidValue:
            return cudaErrorInvalidValue;
//         case hipErrorJitCompilerNotFound:
//             return cudaErrorJitCompilerNotFound;
        case hipErrorLaunchFailure:
            return cudaErrorLaunchFailure;
//         case hipErrorLaunchFileScopedSurf:
//             return cudaErrorLaunchFileScopedSurf;
//         case hipErrorLaunchFileScopedTex:
//             return cudaErrorLaunchFileScopedTex;
//         case hipErrorLaunchIncompatibleTexturing:
//             return cudaErrorLaunchIncompatibleTexturing;
//         case hipErrorLaunchMaxDepthExceeded:
//             return cudaErrorLaunchMaxDepthExceeded;
        case hipErrorLaunchOutOfResources:
            return cudaErrorLaunchOutOfResources;
//         case hipErrorLaunchPendingCountExceeded:
//             return cudaErrorLaunchPendingCountExceeded;
        case hipErrorLaunchTimeOut:
            return cudaErrorLaunchTimeout;
        case hipErrorMapFailed:
            return cudaErrorMapBufferObjectFailed;
        case hipErrorOutOfMemory:
            return cudaErrorMemoryAllocation;
//         case hipErrorMemoryValueTooLarge:
//             return cudaErrorMemoryValueTooLarge;
//         case hipErrorMisalignedAddress:
//             return cudaErrorMisalignedAddress;
        case hipErrorMissingConfiguration:
            return cudaErrorMissingConfiguration;
//         case hipErrorMixedDeviceExecution:
//             return cudaErrorMixedDeviceExecution;
        case hipErrorNoDevice:
            return cudaErrorNoDevice;
        case hipErrorNoBinaryForGpu:
            return cudaErrorNoKernelImageForDevice;
        case hipErrorNotMapped:
            return cudaErrorNotMapped;
        case hipErrorNotMappedAsArray:
            return cudaErrorNotMappedAsArray;
        case hipErrorNotMappedAsPointer:
            return cudaErrorNotMappedAsPointer;
//         case hipErrorNotPermitted:
//             return cudaErrorNotPermitted;
        case hipErrorNotReady:
            return cudaErrorNotReady;
        case hipErrorNotSupported:
            return cudaErrorNotSupported;
//         case hipErrorNotYetImplemented:
//             return cudaErrorNotYetImplemented;
//         case hipErrorNvlinkUncorrectable:
//             return cudaErrorNvlinkUncorrectable;
        case hipErrorOperatingSystem:
            return cudaErrorOperatingSystem;
        case hipErrorPeerAccessAlreadyEnabled:
            return cudaErrorPeerAccessAlreadyEnabled;
        case hipErrorPeerAccessNotEnabled:
            return cudaErrorPeerAccessNotEnabled;
        case hipErrorPeerAccessUnsupported:
            return cudaErrorPeerAccessUnsupported;
        case hipErrorPriorLaunchFailure:
            return cudaErrorPriorLaunchFailure;
        case hipErrorProfilerAlreadyStarted:
            return cudaErrorProfilerAlreadyStarted;
        case hipErrorProfilerAlreadyStopped:
            return cudaErrorProfilerAlreadyStopped;
        case hipErrorProfilerDisabled:
            return cudaErrorProfilerDisabled;
        case hipErrorProfilerNotInitialized:
            return cudaErrorProfilerNotInitialized;
        case hipErrorSetOnActiveProcess:
            return cudaErrorSetOnActiveProcess;
        case hipErrorSharedObjectInitFailed:
            return cudaErrorSharedObjectInitFailed;
        case hipErrorSharedObjectSymbolNotFound:
            return cudaErrorSharedObjectSymbolNotFound;
//         case hipErrorStartupFailure:
//             return cudaErrorStartupFailure;
        case hipErrorStreamCaptureImplicit:
            return cudaErrorStreamCaptureImplicit;
        case hipErrorStreamCaptureInvalidated:
            return cudaErrorStreamCaptureInvalidated;
        case hipErrorStreamCaptureIsolation:
            return cudaErrorStreamCaptureIsolation;
        case hipErrorStreamCaptureMerge:
            return cudaErrorStreamCaptureMerge;
        case hipErrorStreamCaptureUnjoined:
            return cudaErrorStreamCaptureUnjoined;
        case hipErrorStreamCaptureUnmatched:
            return cudaErrorStreamCaptureUnmatched;
        case hipErrorStreamCaptureUnsupported:
            return cudaErrorStreamCaptureUnsupported;
        case hipErrorStreamCaptureWrongThread:
            return cudaErrorStreamCaptureWrongThread;
        case hipErrorNotFound:
            return cudaErrorSymbolNotFound;
//         case hipErrorSyncDepthExceeded:
//             return cudaErrorSyncDepthExceeded;
//         case hipErrorSynchronizationError:
//             return cudaErrorSynchronizationError;
//         case hipErrorSystemDriverMismatch:
//             return cudaErrorSystemDriverMismatch;
//         case hipErrorSystemNotReady:
//             return cudaErrorSystemNotReady;
//         case hipErrorTextureFetchFailed:
//             return cudaErrorTextureFetchFailed;
//         case hipErrorTextureNotBound:
//             return cudaErrorTextureNotBound;
//         case hipErrorTimeout:
//             return cudaErrorTimeout;
//         case hipErrorTooManyPeers:
//             return cudaErrorTooManyPeers;
        case hipErrorUnknown:
        case hipErrorRuntimeOther:
        case hipErrorTbd:
            return cudaErrorUnknown;
        case hipErrorUnmapFailed:
            return cudaErrorUnmapBufferObjectFailed;
        case hipErrorUnsupportedLimit:
            return cudaErrorUnsupportedLimit;
        case hipSuccess:
            return cudaSuccess;
        default:
            ERROR_INVALID_ENUM();
    }
}

hipMemcpyKind cudaMemcpyKindTohipMemcpyKind(enum cudaMemcpyKind para) {
    switch (para) {
        case cudaMemcpyDefault:
            return hipMemcpyDefault;
        case cudaMemcpyDeviceToDevice:
            return hipMemcpyDeviceToDevice;
        case cudaMemcpyDeviceToHost:
            return hipMemcpyDeviceToHost;
        case cudaMemcpyHostToDevice:
            return hipMemcpyHostToDevice;
        case cudaMemcpyHostToHost:
            return hipMemcpyHostToHost;
        default:
            ERROR_INVALID_ENUM();
    }
}

NppStatus hipRppStatusTocudaNppStatus(RppStatus para)
{
	switch (para) {
        case RPP_SUCCESS:
            return NPP_NO_ERROR;
        case RPP_ERROR:
            return NPP_ERROR;
        case RPP_ERROR_INVALID_ARGUMENTS:
            return NPP_BAD_ARGUMENT_ERROR;
		case RPP_ERROR_LOW_OFFSET:
            return NPP_SIZE_ERROR;
		case RPP_ERROR_ZERO_DIVISION:
            return NPP_DIVIDE_BY_ZERO_ERROR;
		case RPP_ERROR_HIGH_SRC_DIMENSION:
            return NPP_SIZE_ERROR;
		case RPP_ERROR_NOT_IMPLEMENTED:
            return NPP_NOT_IMPLEMENTED_ERROR;
		case RPP_ERROR_INVALID_SRC_CHANNELS:
            return NPP_CHANNEL_ERROR;
		case RPP_ERROR_INVALID_DST_CHANNELS:
            return NPP_CHANNEL_ERROR;
		case RPP_ERROR_INVALID_SRC_LAYOUT:
            return NPP_WRONG_INTERSECTION_ROI_ERROR;
		case RPP_ERROR_INVALID_DST_LAYOUT:
            return NPP_WRONG_INTERSECTION_ROI_ERROR;
		case RPP_ERROR_INVALID_SRC_DATATYPE:
            return NPP_DATA_TYPE_ERROR;
		case RPP_ERROR_INVALID_DST_DATATYPE:
            return NPP_DATA_TYPE_ERROR;
		case RPP_ERROR_INVALID_SRC_OR_DST_DATATYPE:
            return NPP_DATA_TYPE_ERROR;
		case RPP_ERROR_INSUFFICIENT_DST_BUFFER_LENGTH:
            return NPP_SIZE_ERROR;
		case RPP_ERROR_INVALID_PARAMETER_DATATYPE:
            return NPP_DATA_TYPE_ERROR;
		case RPP_ERROR_NOT_ENOUGH_MEMORY:
            return NPP_NO_MEMORY_ERROR;
		case RPP_ERROR_OUT_OF_BOUND_SRC_ROI:
            return NPP_OUT_OFF_RANGE_ERROR;
		case RPP_ERROR_SRC_DST_LAYOUT_MISMATCH:
            return NPP_WRONG_INTERSECTION_ROI_ERROR;
		case RPP_ERROR_INVALID_CHANNELS:
            return NPP_CHANNEL_ERROR;
        default:
            ERROR_INVALID_ENUM();
    }
}

__host__ cudaError cudaMalloc(void ** devPtr,size_t size)
{
    hipError_t hip_res;
    hip_res = hipMalloc(devPtr, size);
    return hipErrorTocudaError(hip_res);
}

__host__ cudaError cudaMemcpy(void * dst,const void * src,size_t count,enum cudaMemcpyKind kind)
{
    hipMemcpyKind hip_kind = cudaMemcpyKindTohipMemcpyKind(kind);
    hipError_t hip_res;
    hip_res = hipMemcpy(dst, src, count, hip_kind);
    return hipErrorTocudaError(hip_res);
}

__host__ cudaError cudaFree(void * devPtr)
{
    hipError_t hip_res;
    hip_res = hipFree(devPtr);
    return hipErrorTocudaError(hip_res);
}

Npp8u* nppiMalloc_8u_C1(int nWidthPixels, int nHeightPixels, int* pStepBytes) 
{  
    size_t imageSize = nWidthPixels * nHeightPixels * sizeof(Npp8u);  
    
    Npp8u* d_image = nullptr;  
    cudaError cudaStatus = cudaMalloc((void**)&d_image, imageSize);  
    
    if (cudaStatus != cudaSuccess) {  
        std::cerr << "Error allocating memory on device" << std::endl;
        return nullptr;  
    }  
    
    if (pStepBytes) {  
        *pStepBytes = nWidthPixels * sizeof(Npp8u); 
    }  
    
    return d_image;  
}  

Npp16s* nppiMalloc_16s_C1(int nWidthPixels, int nHeightPixels, int* pStepBytes)
{
    size_t imageSize = nWidthPixels * nHeightPixels * sizeof(Npp16s);

    Npp16s* d_image = nullptr;
    cudaError cudaStatus = cudaMalloc((void**)&d_image, imageSize);

    if (cudaStatus != cudaSuccess) {
        std::cerr << "Error allocating memory on device" << std::endl;
        return nullptr;
    }

    if (pStepBytes) {
        *pStepBytes = nWidthPixels * sizeof(Npp16s);
    }

    return d_image;
}

void nppiFree(void * pData)
{
   cudaError cudaStatus = cudaFree(pData);
   if (cudaStatus != cudaSuccess) {
        std::cerr << "Error Free memory on device" << std::endl;     
   }
}

const NppLibraryVersion* nppGetLibVersion() {  
    static NppLibraryVersion version = {11, 3, 6};
    return &version;  
}

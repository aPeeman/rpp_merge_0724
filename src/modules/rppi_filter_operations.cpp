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
#include "rppi_filter_operations.h"
#include "cpu/host_filter_operations.hpp"

#ifdef HIP_COMPILE
#include "rpp_hip_common.hpp"
#include "hip/hip_declarations.hpp"
#elif defined(OCL_COMPILE)
#include "rpp_cl_common.hpp"
#include "cl/cl_declarations.hpp"
#endif //backend

/******************** box_filter ********************/

RppStatus
rppi_box_filter_u8_pln1_batchPD_host(RppPtr_t srcPtr,
                                     RppiSize *srcSize,
                                     RppiSize maxSrcSize,
                                     RppPtr_t dstPtr,
                                     Rpp32u *kernelSize,
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

    box_filter_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                                 srcSize,
                                 rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                 static_cast<Rpp8u*>(dstPtr),
                                 kernelSize,
                                 rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                                 rpp::deref(rppHandle).GetBatchSize(),
                                 RPPI_CHN_PLANAR,
                                 1,
                                 rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_box_filter_u8_pln3_batchPD_host(RppPtr_t srcPtr,
                                     RppiSize *srcSize,
                                     RppiSize maxSrcSize,
                                     RppPtr_t dstPtr,
                                     Rpp32u *kernelSize,
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

    box_filter_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                                 srcSize,
                                 rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                 static_cast<Rpp8u*>(dstPtr),
                                 kernelSize,
                                 rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                                 rpp::deref(rppHandle).GetBatchSize(),
                                 RPPI_CHN_PLANAR,
                                 3,
                                 rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_box_filter_u8_pkd3_batchPD_host(RppPtr_t srcPtr,
                                     RppiSize *srcSize,
                                     RppiSize maxSrcSize,
                                     RppPtr_t dstPtr,
                                     Rpp32u *kernelSize,
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

    box_filter_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                                 srcSize,
                                 rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                 static_cast<Rpp8u*>(dstPtr),
                                 kernelSize,
                                 rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                                 rpp::deref(rppHandle).GetBatchSize(),
                                 RPPI_CHN_PACKED,
                                 3,
                                 rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

/******************** sobel_filter ********************/

RppStatus
rppi_sobel_filter_u8_pln1_batchPD_host(RppPtr_t srcPtr,
                                       RppiSize *srcSize,
                                       RppiSize maxSrcSize,
                                       RppPtr_t dstPtr,
                                       Rpp32u *sobelType,
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

    sobel_filter_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                                   srcSize,
                                   rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                   static_cast<Rpp8u*>(dstPtr),
                                   sobelType,
                                   rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                                   rpp::deref(rppHandle).GetBatchSize(),
                                   RPPI_CHN_PLANAR,
                                   1,
                                   rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_sobel_filter_u8_pln3_batchPD_host(RppPtr_t srcPtr,
                                       RppiSize *srcSize,
                                       RppiSize maxSrcSize,
                                       RppPtr_t dstPtr,
                                       Rpp32u *sobelType,
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

    sobel_filter_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                                   srcSize,
                                   rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                   static_cast<Rpp8u*>(dstPtr),
                                   sobelType,
                                   rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                                   rpp::deref(rppHandle).GetBatchSize(),
                                   RPPI_CHN_PLANAR,
                                   3,
                                   rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_sobel_filter_u8_pkd3_batchPD_host(RppPtr_t srcPtr,
                                       RppiSize *srcSize,
                                       RppiSize maxSrcSize,
                                       RppPtr_t dstPtr,
                                       Rpp32u *sobelType,
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

    sobel_filter_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                                   srcSize,
                                   rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                   static_cast<Rpp8u*>(dstPtr),
                                   sobelType,
                                   rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                                   rpp::deref(rppHandle).GetBatchSize(),
                                   RPPI_CHN_PACKED,
                                   3,
                                   rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

NppStatus nppiFilterSobelHoriz_8u_C3R(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep, NppiSize oSizeROI)
{
        int noOfImages = 1;
        int ip_channel = 3;//pkd_3
        RppiSize *srcSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize));
        RppiSize maxSize;
        srcSize->width  = oSizeROI.width;
        srcSize->height = oSizeROI.height;
        maxSize.width  = oSizeROI.width;
        maxSize.height = oSizeROI.height;
	    //Rpp32u kernelSize = oMaskSize.width;
	    Rpp32u sobelType = 1;//1 for Horiz,0 for Vert

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
	    status = rppi_sobel_filter_u8_pkd3_batchPD_gpu((RppPtr_t)pSrc, srcSize, maxSize, (RppPtr_t)pDst, &sobelType, noOfImages, handle);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        free(srcSize);

        return(hipRppStatusTocudaNppStatus(status));
}										

NppStatus nppiFilterSobelHoriz_8u_C1R_Ctx(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx)
{
        int noOfImages = 1;
        int ip_channel = 1;//pln1
        RppiSize *srcSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize));
        RppiSize maxSize;
        srcSize->width  = oSizeROI.width;
        srcSize->height = oSizeROI.height;
        maxSize.width  = oSizeROI.width;
        maxSize.height = oSizeROI.height;
        //Rpp32u kernelSize = oMaskSize.width;
        Rpp32u sobelType = 1;//1 for Horiz,0 for Vert

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        status = rppi_sobel_filter_u8_pln1_batchPD_gpu((RppPtr_t)pSrc, srcSize, maxSize, (RppPtr_t)pDst, &sobelType, noOfImages, handle);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        free(srcSize);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiFilterSobelHoriz_8u_C1R(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep, NppiSize oSizeROI)
{
        int noOfImages = 1;
        int ip_channel = 1;//pln1
        RppiSize *srcSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize)); 
        RppiSize maxSize;
        srcSize->width  = oSizeROI.width;
        srcSize->height = oSizeROI.height;
        maxSize.width  = oSizeROI.width;
        maxSize.height = oSizeROI.height;
        //Rpp32u kernelSize = oMaskSize.width;
        Rpp32u sobelType = 1;//1 for Horiz,0 for Vert
        
        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        status = rppi_sobel_filter_u8_pln1_batchPD_gpu((RppPtr_t)pSrc, srcSize, maxSize, (RppPtr_t)pDst, &sobelType, noOfImages, handle);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        free(srcSize);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiFilterSobelHoriz_8u_C3R_Ctx(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx)
{
        int noOfImages = 1;
        int ip_channel = 3;//pkd_3
        RppiSize *srcSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize));
        RppiSize maxSize;
        srcSize->width  = oSizeROI.width;
        srcSize->height = oSizeROI.height;
        maxSize.width  = oSizeROI.width;
        maxSize.height = oSizeROI.height;
        //Rpp32u kernelSize = oMaskSize.width;
        Rpp32u sobelType = 1;//1 for Horiz,0 for Vert

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        status = rppi_sobel_filter_u8_pkd3_batchPD_gpu((RppPtr_t)pSrc, srcSize, maxSize, (RppPtr_t)pDst, &sobelType, noOfImages, handle);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        free(srcSize);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiFilterSobelVert_8u_C3R(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep, NppiSize oSizeROI)
{
        int noOfImages = 1;
        int ip_channel = 3;//pkd_3
        RppiSize *srcSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize));
        RppiSize maxSize;               
        srcSize->width  = oSizeROI.width;
        srcSize->height = oSizeROI.height;
        maxSize.width  = oSizeROI.width;
        maxSize.height = oSizeROI.height;
        //Rpp32u kernelSize = oMaskSize.width;
        Rpp32u sobelType = 0;//1 for Horiz,0 for Vert

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        status = rppi_sobel_filter_u8_pkd3_batchPD_gpu((RppPtr_t)pSrc, srcSize, maxSize, (RppPtr_t)pDst, &sobelType, noOfImages, handle);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        free(srcSize);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiFilterSobelVert_8u_C1R_Ctx(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx)
{
        int noOfImages = 1;
        int ip_channel = 1;//pln1
        RppiSize *srcSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize));
        RppiSize maxSize;
        srcSize->width  = oSizeROI.width;
        srcSize->height = oSizeROI.height;
        maxSize.width  = oSizeROI.width;
        maxSize.height = oSizeROI.height;
        //Rpp32u kernelSize = oMaskSize.width;
        Rpp32u sobelType = 0;//1 for Horiz,0 for Vert

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        status = rppi_sobel_filter_u8_pln1_batchPD_gpu((RppPtr_t)pSrc, srcSize, maxSize, (RppPtr_t)pDst, &sobelType, noOfImages, handle);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        free(srcSize);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiFilterSobelVert_8u_C1R(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep, NppiSize oSizeROI)
{
        int noOfImages = 1;
        int ip_channel = 1;//pln1
        RppiSize *srcSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize));
        RppiSize maxSize;
        srcSize->width  = oSizeROI.width;
        srcSize->height = oSizeROI.height;
        maxSize.width  = oSizeROI.width;
        maxSize.height = oSizeROI.height;
        //Rpp32u kernelSize = oMaskSize.width;
        Rpp32u sobelType = 0;//1 for Horiz,0 for Vert
                                        
        rppHandle_t handle;             
        hipStream_t stream;             
        hipStreamCreate(&stream);       
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        status = rppi_sobel_filter_u8_pln1_batchPD_gpu((RppPtr_t)pSrc, srcSize, maxSize, (RppPtr_t)pDst, &sobelType, noOfImages, handle);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        free(srcSize);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiFilterSobelVert_8u_C3R_Ctx(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx)
{
        int noOfImages = 1;
        int ip_channel = 3;//pkd_3
        RppiSize *srcSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize));
        RppiSize maxSize;
        srcSize->width  = oSizeROI.width;
        srcSize->height = oSizeROI.height;
        maxSize.width  = oSizeROI.width;
        maxSize.height = oSizeROI.height;
        //Rpp32u kernelSize = oMaskSize.width;
        Rpp32u sobelType = 0;//1 for Horiz,0 for Vert

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        status = rppi_sobel_filter_u8_pkd3_batchPD_gpu((RppPtr_t)pSrc, srcSize, maxSize, (RppPtr_t)pDst, &sobelType, noOfImages, handle);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        free(srcSize);

        return(hipRppStatusTocudaNppStatus(status));
}

/******************** median_filter ********************/

RppStatus
rppi_median_filter_u8_pln1_batchPD_host(RppPtr_t srcPtr,
                                        RppiSize *srcSize,
                                        RppiSize maxSrcSize,
                                        RppPtr_t dstPtr,
                                        Rpp32u *kernelSize,
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

    median_filter_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                                    srcSize,
                                    rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                    static_cast<Rpp8u*>(dstPtr),
                                    kernelSize,
                                    rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                                    rpp::deref(rppHandle).GetBatchSize(),
                                    RPPI_CHN_PLANAR,
                                    1,
                                    rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_median_filter_u8_pln3_batchPD_host(RppPtr_t srcPtr,
                                        RppiSize *srcSize,
                                        RppiSize maxSrcSize,
                                        RppPtr_t dstPtr,
                                        Rpp32u *kernelSize,
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

    median_filter_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                                    srcSize,
                                    rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                    static_cast<Rpp8u*>(dstPtr),
                                    kernelSize,
                                    rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                                    rpp::deref(rppHandle).GetBatchSize(),
                                    RPPI_CHN_PLANAR,
                                    3,
                                    rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_median_filter_u8_pkd3_batchPD_host(RppPtr_t srcPtr,
                                        RppiSize *srcSize,
                                        RppiSize maxSrcSize,
                                        RppPtr_t dstPtr,
                                        Rpp32u *kernelSize,
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

    median_filter_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                                    srcSize,
                                    rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                    static_cast<Rpp8u*>(dstPtr),
                                    kernelSize,
                                    rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                                    rpp::deref(rppHandle).GetBatchSize(),
                                    RPPI_CHN_PACKED,
                                    3,
                                    rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

/******************** non_max_suppression ********************/

RppStatus
rppi_non_max_suppression_u8_pln1_batchPD_host(RppPtr_t srcPtr,
                                              RppiSize *srcSize,
                                              RppiSize maxSrcSize,
                                              RppPtr_t dstPtr,
                                              Rpp32u *kernelSize,
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

    non_max_suppression_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                                          srcSize,
                                          rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                          static_cast<Rpp8u*>(dstPtr),
                                          kernelSize,
                                          rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                                          rpp::deref(rppHandle).GetBatchSize(),
                                          RPPI_CHN_PLANAR,
                                          1,
                                          rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_non_max_suppression_u8_pln3_batchPD_host(RppPtr_t srcPtr,
                                              RppiSize *srcSize,
                                              RppiSize maxSrcSize,
                                              RppPtr_t dstPtr,
                                              Rpp32u *kernelSize,
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

    non_max_suppression_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                                          srcSize,
                                          rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                          static_cast<Rpp8u*>(dstPtr),
                                          kernelSize,
                                          rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                                          rpp::deref(rppHandle).GetBatchSize(),
                                          RPPI_CHN_PLANAR,
                                          3,
                                          rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_non_max_suppression_u8_pkd3_batchPD_host(RppPtr_t srcPtr,
                                              RppiSize *srcSize,
                                              RppiSize maxSrcSize,
                                              RppPtr_t dstPtr,
                                              Rpp32u *kernelSize,
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

    non_max_suppression_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                                          srcSize,
                                          rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                          static_cast<Rpp8u*>(dstPtr),
                                          kernelSize,
                                          rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                                          rpp::deref(rppHandle).GetBatchSize(),
                                          RPPI_CHN_PACKED,
                                          3,
                                          rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

/******************** gaussian_filter ********************/

RppStatus
rppi_gaussian_filter_u8_pln1_batchPD_host(RppPtr_t srcPtr,
                                          RppiSize *srcSize,
                                          RppiSize maxSrcSize,
                                          RppPtr_t dstPtr,
                                          Rpp32f *stdDev,
                                          Rpp32u *kernelSize,
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

    gaussian_filter_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                                      srcSize,
                                      rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                      static_cast<Rpp8u*>(dstPtr),
                                      stdDev,
                                      kernelSize,
                                      rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                                      rpp::deref(rppHandle).GetBatchSize(),
                                      RPPI_CHN_PLANAR,
                                      1,
                                      rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_gaussian_filter_u8_pln3_batchPD_host(RppPtr_t srcPtr,
                                          RppiSize *srcSize,
                                          RppiSize maxSrcSize,
                                          RppPtr_t dstPtr,
                                          Rpp32f *stdDev,
                                          Rpp32u *kernelSize,
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

    gaussian_filter_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                                      srcSize,
                                      rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                      static_cast<Rpp8u*>(dstPtr),
                                      stdDev,
                                      kernelSize,
                                      rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                                      rpp::deref(rppHandle).GetBatchSize(),
                                      RPPI_CHN_PLANAR,
                                      3,
                                      rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_gaussian_filter_u8_pkd3_batchPD_host(RppPtr_t srcPtr,
                                          RppiSize *srcSize,
                                          RppiSize maxSrcSize,
                                          RppPtr_t dstPtr,
                                          Rpp32f *stdDev,
                                          Rpp32u *kernelSize,
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

    gaussian_filter_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                                      srcSize,
                                      rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                      static_cast<Rpp8u*>(dstPtr),
                                      stdDev,
                                      kernelSize,
                                      rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                                      rpp::deref(rppHandle).GetBatchSize(),
                                      RPPI_CHN_PACKED,
                                      3,
                                      rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

NppStatus nppiFilterGauss_8u_C1R_Ctx(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiMaskSize eMaskSize, NppStreamContext nppStreamCtx)
{
        unsigned int kernelSize;
        float stdDev;
        RppStatus status;
        if((eMaskSize == NPP_MASK_SIZE_1_X_3) || (eMaskSize == NPP_MASK_SIZE_1_X_5) || (eMaskSize == NPP_MASK_SIZE_3_X_1) || (eMaskSize == NPP_MASK_SIZE_5_X_1))
        {
                status = RPP_ERROR_INVALID_ARGUMENTS;
                return(hipRppStatusTocudaNppStatus(status));
        }else if(eMaskSize ==  NPP_MASK_SIZE_3_X_3)
        {
                kernelSize = 3;
                stdDev = 0.4 + (kernelSize/2) * 0.6;
        }else if(eMaskSize == NPP_MASK_SIZE_5_X_5)
        {
                kernelSize = 5;
                stdDev = 0.4 + (kernelSize/2) * 0.6;
        }else if(eMaskSize == NPP_MASK_SIZE_7_X_7)
        {
                kernelSize = 7;
                stdDev = 0.4 + (kernelSize/2) * 0.6;
        }else if(eMaskSize == NPP_MASK_SIZE_9_X_9)
        {
                kernelSize = 9;
                stdDev = 0.4 + (kernelSize/2) * 0.6;
        }else if(eMaskSize == NPP_MASK_SIZE_11_X_11)
        {
                kernelSize = 11;
                stdDev = 0.4 + (kernelSize/2) * 0.6;
        }else if(eMaskSize == NPP_MASK_SIZE_13_X_13)
        {
                kernelSize = 13;
                stdDev = 0.4 + (kernelSize/2) * 0.6;
        }else if(eMaskSize == NPP_MASK_SIZE_15_X_15)
        {
                kernelSize = 15;
                stdDev = 0.4 + (kernelSize/2) * 0.6;
        }
        
        int noOfImages = 1;
        int ip_channel = 1;//pln1
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

	    status = rppi_gaussian_filter_u8_pln1_batchPD_gpu((RppPtr_t)pSrc, srcSize, maxSize, (RppPtr_t)pDst, &stdDev, &kernelSize, noOfImages, handle);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        free(srcSize);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiFilterGauss_8u_C1R(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiMaskSize eMaskSize)
{
        unsigned int kernelSize;
        float stdDev;
        RppStatus status;
        if((eMaskSize == NPP_MASK_SIZE_1_X_3) || (eMaskSize == NPP_MASK_SIZE_1_X_5) || (eMaskSize == NPP_MASK_SIZE_3_X_1) || (eMaskSize == NPP_MASK_SIZE_5_X_1))
        {
                status = RPP_ERROR_INVALID_ARGUMENTS;
                return(hipRppStatusTocudaNppStatus(status));
        }else if(eMaskSize ==  NPP_MASK_SIZE_3_X_3)
        {
                kernelSize = 3;
                stdDev = 0.4 + (kernelSize/2) * 0.6;
        }else if(eMaskSize == NPP_MASK_SIZE_5_X_5)
        {
                kernelSize = 5;
                stdDev = 0.4 + (kernelSize/2) * 0.6;
        }else if(eMaskSize == NPP_MASK_SIZE_7_X_7)
        {
                kernelSize = 7;
                stdDev = 0.4 + (kernelSize/2) * 0.6;
        }else if(eMaskSize == NPP_MASK_SIZE_9_X_9)
        {
                kernelSize = 9;
                stdDev = 0.4 + (kernelSize/2) * 0.6;
        }else if(eMaskSize == NPP_MASK_SIZE_11_X_11)
        {
                kernelSize = 11;
                stdDev = 0.4 + (kernelSize/2) * 0.6;
        }else if(eMaskSize == NPP_MASK_SIZE_13_X_13)
        {
                kernelSize = 13;
                stdDev = 0.4 + (kernelSize/2) * 0.6;
        }else if(eMaskSize == NPP_MASK_SIZE_15_X_15)
        {
                kernelSize = 15;
                stdDev = 0.4 + (kernelSize/2) * 0.6;
        }
        
        int noOfImages = 1;
        int ip_channel = 1;//pln1
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

	    status = rppi_gaussian_filter_u8_pln1_batchPD_gpu((RppPtr_t)pSrc, srcSize, maxSize, (RppPtr_t)pDst, &stdDev, &kernelSize, noOfImages, handle);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        free(srcSize);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiFilterGauss_8u_C3R_Ctx(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiMaskSize eMaskSize, NppStreamContext nppStreamCtx)
{
        unsigned int kernelSize;
        float stdDev;
        RppStatus status;
        if((eMaskSize == NPP_MASK_SIZE_1_X_3) || (eMaskSize == NPP_MASK_SIZE_1_X_5) || (eMaskSize == NPP_MASK_SIZE_3_X_1) || (eMaskSize == NPP_MASK_SIZE_5_X_1))
        {
                status = RPP_ERROR_INVALID_ARGUMENTS;
                return(hipRppStatusTocudaNppStatus(status));
        }else if(eMaskSize ==  NPP_MASK_SIZE_3_X_3)
        {
                kernelSize = 3;
                stdDev = 0.4 + (kernelSize/2) * 0.6;
        }else if(eMaskSize == NPP_MASK_SIZE_5_X_5)
        {
                kernelSize = 5;
                stdDev = 0.4 + (kernelSize/2) * 0.6;
        }else if(eMaskSize == NPP_MASK_SIZE_7_X_7)
        {
                kernelSize = 7;
                stdDev = 0.4 + (kernelSize/2) * 0.6;
        }else if(eMaskSize == NPP_MASK_SIZE_9_X_9)
        {
                kernelSize = 9;
                stdDev = 0.4 + (kernelSize/2) * 0.6;
        }else if(eMaskSize == NPP_MASK_SIZE_11_X_11)
        {
                kernelSize = 11;
                stdDev = 0.4 + (kernelSize/2) * 0.6;
        }else if(eMaskSize == NPP_MASK_SIZE_13_X_13)
        {
                kernelSize = 13;
                stdDev = 0.4 + (kernelSize/2) * 0.6;
        }else if(eMaskSize == NPP_MASK_SIZE_15_X_15)
        {
                kernelSize = 15;
                stdDev = 0.4 + (kernelSize/2) * 0.6;
        }
        
        int noOfImages = 1;
        int ip_channel = 3;//pkd_3
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

	    status = rppi_gaussian_filter_u8_pkd3_batchPD_gpu((RppPtr_t)pSrc, srcSize, maxSize, (RppPtr_t)pDst, &stdDev, &kernelSize, noOfImages, handle);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        free(srcSize);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiFilterGauss_8u_C3R(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiMaskSize eMaskSize)
{
        unsigned int kernelSize;
        float stdDev;
        RppStatus status;
        if((eMaskSize == NPP_MASK_SIZE_1_X_3) || (eMaskSize == NPP_MASK_SIZE_1_X_5) || (eMaskSize == NPP_MASK_SIZE_3_X_1) || (eMaskSize == NPP_MASK_SIZE_5_X_1))
        {
                status = RPP_ERROR_INVALID_ARGUMENTS;
                return(hipRppStatusTocudaNppStatus(status));
        }else if(eMaskSize ==  NPP_MASK_SIZE_3_X_3)
        {
                kernelSize = 3;
                stdDev = 0.4 + (kernelSize/2) * 0.6;
        }else if(eMaskSize == NPP_MASK_SIZE_5_X_5)
        {
                kernelSize = 5;
                stdDev = 0.4 + (kernelSize/2) * 0.6;
        }else if(eMaskSize == NPP_MASK_SIZE_7_X_7)
        {
                kernelSize = 7;
                stdDev = 0.4 + (kernelSize/2) * 0.6;
        }else if(eMaskSize == NPP_MASK_SIZE_9_X_9)
        {
                kernelSize = 9;
                stdDev = 0.4 + (kernelSize/2) * 0.6;
        }else if(eMaskSize == NPP_MASK_SIZE_11_X_11)
        {
                kernelSize = 11;
                stdDev = 0.4 + (kernelSize/2) * 0.6;
        }else if(eMaskSize == NPP_MASK_SIZE_13_X_13)
        {
                kernelSize = 13;
                stdDev = 0.4 + (kernelSize/2) * 0.6;
        }else if(eMaskSize == NPP_MASK_SIZE_15_X_15)
        {
                kernelSize = 15;
                stdDev = 0.4 + (kernelSize/2) * 0.6;
        }
        
        int noOfImages = 1;
        int ip_channel = 3;//pkd_3
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

	    status = rppi_gaussian_filter_u8_pkd3_batchPD_gpu((RppPtr_t)pSrc, srcSize, maxSize, (RppPtr_t)pDst, &stdDev, &kernelSize, noOfImages, handle);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        free(srcSize);

        return(hipRppStatusTocudaNppStatus(status));
}

/******************** nonlinear_filter ********************/

RppStatus
rppi_nonlinear_filter_u8_pln1_batchPD_host(RppPtr_t srcPtr,
                                           RppiSize *srcSize,
                                           RppiSize maxSrcSize,
                                           RppPtr_t dstPtr,
                                           Rpp32u *kernelSize,
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

    median_filter_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                                    srcSize,
                                    rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                    static_cast<Rpp8u*>(dstPtr),
                                    kernelSize,
                                    rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                                    rpp::deref(rppHandle).GetBatchSize(),
                                    RPPI_CHN_PLANAR,
                                    1,
                                    rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_nonlinear_filter_u8_pln3_batchPD_host(RppPtr_t srcPtr,
                                           RppiSize *srcSize,
                                           RppiSize maxSrcSize,
                                           RppPtr_t dstPtr,
                                           Rpp32u *kernelSize,
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

    median_filter_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                                    srcSize,
                                    rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                    static_cast<Rpp8u*>(dstPtr),
                                    kernelSize,
                                    rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                                    rpp::deref(rppHandle).GetBatchSize(),
                                    RPPI_CHN_PLANAR,
                                    3,
                                    rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_nonlinear_filter_u8_pkd3_batchPD_host(RppPtr_t srcPtr,
                                           RppiSize *srcSize,
                                           RppiSize maxSrcSize,
                                           RppPtr_t dstPtr,
                                           Rpp32u *kernelSize,
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

    median_filter_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                                    srcSize,
                                    rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                    static_cast<Rpp8u*>(dstPtr),
                                    kernelSize,
                                    rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                                    rpp::deref(rppHandle).GetBatchSize(),
                                    RPPI_CHN_PACKED,
                                    3,
                                    rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

/******************** custom_convolution ********************/

RppStatus
rppi_custom_convolution_u8_pln1_batchPD_host(RppPtr_t srcPtr,
                                             RppiSize *srcSize,
                                             RppiSize maxSrcSize,
                                             RppPtr_t dstPtr,
                                             RppPtr_t kernel,
                                             RppiSize *kernelSize,
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

    custom_convolution_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                                         srcSize,
                                         rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                         static_cast<Rpp8u*>(dstPtr),
                                         static_cast<Rpp32f*>(kernel),
                                         kernelSize,
                                         rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                                         rpp::deref(rppHandle).GetBatchSize(),
                                         RPPI_CHN_PLANAR,
                                         1,
                                         rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_custom_convolution_u8_pln3_batchPD_host(RppPtr_t srcPtr,
                                             RppiSize *srcSize,
                                             RppiSize maxSrcSize,
                                             RppPtr_t dstPtr,
                                             RppPtr_t kernel,
                                             RppiSize *kernelSize,
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

    custom_convolution_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                                         srcSize,
                                         rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                         static_cast<Rpp8u*>(dstPtr),
                                         static_cast<Rpp32f*>(kernel),
                                         kernelSize,
                                         rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                                         rpp::deref(rppHandle).GetBatchSize(),
                                         RPPI_CHN_PLANAR,
                                         3,
                                         rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_custom_convolution_u8_pkd3_batchPD_host(RppPtr_t srcPtr,
                                             RppiSize *srcSize,
                                             RppiSize maxSrcSize,
                                             RppPtr_t dstPtr,
                                             RppPtr_t kernel,
                                             RppiSize *kernelSize,
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

    custom_convolution_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                                         srcSize,
                                         rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                         static_cast<Rpp8u*>(dstPtr),
                                         static_cast<Rpp32f*>(kernel),
                                         kernelSize,
                                         rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                                         rpp::deref(rppHandle).GetBatchSize(),
                                         RPPI_CHN_PACKED,
                                         3,
                                         rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

/********************************************************************************************************************/
/*********************************************** RPP_GPU_SUPPORT = ON ***********************************************/
/********************************************************************************************************************/

/*NppStatus nppiFilter_8u_C3R(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep, 
			NppiSize oSizeROI, const Npp32s *pKernel, NppiSize oKernelSize, NppiPoint oAnchor, Npp32s nDivisor)
{
        int noOfImages = 1;
        int ip_channel = 3;
	//RppiChnFormat chnFormat = RPPI_CHN_PACKED;
	//Rpp32u type = 3;
	Rpp32u nbatchSize = 1;
        RppiSize *srcSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize));
	RppiSize *kernelSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize));
        RppiSize maxSize;
        srcSize->width  = oSizeROI.width;
        srcSize->height = oSizeROI.height;
        maxSize.width  = oSizeROI.width;
        maxSize.height = oSizeROI.height;
	kernelSize.width = oKernelSize.width;
	kernelSize.height = oKernelSize.height;

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);
        clock_t start, end;
        double gpu_time_used;

        RppStatus status;
        start = clock();
	status = rppi_custom_convolution_u8_pkd3_batchPD_gpu((RppPtr_t)pSrc, srcSize, maxSize, (RppPtr_t)pDst, (RppPtr_t)pKernel, kernelSize, nbatchSize, handle);
        hipDeviceSynchronize();
        end = clock();

        printf("\nnppiFilter_8u_C3R is %d\n", status);
        gpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
        std::cout << "\nGPU Filter_8u -  : " << gpu_time_used;
        printf("\n");
        rppDestroyGPU(handle);
        free(srcSize);
	free(kernelSize);

        return(hipRppStatusTocudaNppStatus(status));
}*/

#ifdef GPU_SUPPORT

/******************** box_filter ********************/

RppStatus
rppi_box_filter_u8_pln1_batchPD_gpu(RppPtr_t srcPtr,
                                    RppiSize *srcSize,
                                    RppiSize maxSrcSize,
                                    RppPtr_t dstPtr,
                                    Rpp32u *kernelSize,
                                    Rpp32u nbatchSize,
                                    rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    roiPoints.x = (maxSrcSize.width - srcSize->width) / 2;
    roiPoints.y = (maxSrcSize.height - srcSize->height) / 2;
    roiPoints.roiHeight = (maxSrcSize.height + srcSize->height) / 2;
    roiPoints.roiWidth = (maxSrcSize.width + srcSize->width) / 2;
    Rpp32u paramIndex = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    copy_roi(roiPoints, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
    copy_param_uint(kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        box_filter_cl_batch(static_cast<cl_mem>(srcPtr),
                            static_cast<cl_mem>(dstPtr),
                            rpp::deref(rppHandle),
                            RPPI_CHN_PLANAR,
                            1);
    }
#elif defined(HIP_COMPILE)
    {
        box_filter_hip_batch(static_cast<Rpp8u*>(srcPtr),
                             static_cast<Rpp8u*>(dstPtr),
                             rpp::deref(rppHandle),
                             RPPI_CHN_PLANAR,
                             1);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

NppStatus nppiFilterBox_8u_C1R(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor)
{
	    int noOfImages = 1;
        int ip_channel = 1;
        RppiSize *srcSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize));
        RppiSize maxSize;
        srcSize->width  = oSizeROI.width;
        srcSize->height = oSizeROI.height;
        maxSize.width  = nSrcStep;
        maxSize.height = nSrcStep;
        Rpp32u kernelSize = oMaskSize.width;

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        status = rppi_box_filter_u8_pln1_batchPD_gpu((RppPtr_t)pSrc, srcSize, maxSize, (RppPtr_t)pDst, &kernelSize, noOfImages, handle);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        free(srcSize);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiFilterBox_8u_C1R_Ctx(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx)
{
        int noOfImages = 1;
        int ip_channel = 1;
        RppiSize *srcSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize));
        RppiSize maxSize;
        srcSize->width  = oSizeROI.width;
        srcSize->height = oSizeROI.height;
        maxSize.width  = nSrcStep;
        maxSize.height = nSrcStep;
        Rpp32u kernelSize = oMaskSize.width;

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        status = rppi_box_filter_u8_pln1_batchPD_gpu((RppPtr_t)pSrc, srcSize, maxSize, (RppPtr_t)pDst, &kernelSize, noOfImages, handle);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        free(srcSize);

        return(hipRppStatusTocudaNppStatus(status));
}

RppStatus
nppi_box_filter_u8_pln1_batchPD_gpu(RppPtr_t srcPtr,
                                    RppiSize *srcSize,
                                    RppiSize maxSrcSize,
                                    RppPtr_t dstPtr,
                                    Rpp32u *kernelSize,
									RppiPoint maskAnchor,
									RppiPoint maskLoc,
                                    Rpp32u nbatchSize,
									RppiBorderType rBorderType,
                                    rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    roiPoints.x = maskLoc.x;
    roiPoints.y = maskLoc.y;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    Rpp32u paramIndex = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    copy_roi(roiPoints, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
    copy_param_uint(kernelSize, rpp::deref(rppHandle), paramIndex++);

    box_filter_npp_batch(static_cast<Rpp8u*>(srcPtr),
                         static_cast<Rpp8u*>(dstPtr),
                         rpp::deref(rppHandle),
                         RPPI_CHN_PLANAR,
                         1,
			             maskAnchor,
			             rBorderType);
	
    return RPP_SUCCESS;
}

NppStatus nppiFilterBoxBorder_8u_C1R(const Npp8u *pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u *pDst, Npp32s nDstStep, 
					NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType)
								
{
    int noOfImages = 1;
    int ip_channel = 1;//pln1
    RppiSize *srcSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize));
    RppiSize maxSize;
	RppiPoint maskLoc, maskAnchor;
	RppiBorderType rBorderType;
    srcSize->width  = oSizeROI.width;
    srcSize->height = oSizeROI.height;
    maxSize.width  = oSrcSize.width;
    maxSize.height = oSrcSize.height;
    Rpp32u kernelSize = oMaskSize.width;
	maskAnchor.x = oAnchor.x;
	maskAnchor.y = oAnchor.y;
	maskLoc.x = oSrcOffset.x;
	maskLoc.y = oSrcOffset.y;
	
	if(eBorderType == NPP_BORDER_UNDEFINED || eBorderType == NPP_BORDER_NONE){
		rBorderType = RPP_BORDER_UNDEFINED;
	} else if(eBorderType == NPP_BORDER_CONSTANT) {
		rBorderType = RPP_BORDER_CONSTANT;
	} else if(eBorderType == NPP_BORDER_REPLICATE) {
		rBorderType = RPP_BORDER_REPLICATE;
	} else if(eBorderType == NPP_BORDER_WRAP) {
		rBorderType = RPP_BORDER_WRAP;
	} else if(eBorderType == NPP_BORDER_MIRROR) {
		rBorderType = RPP_BORDER_MIRROR;
	}
	
    rppHandle_t handle;
    hipStream_t stream;
    hipStreamCreate(&stream);
    rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

    RppStatus status;
    status = nppi_box_filter_u8_pln1_batchPD_gpu((RppPtr_t)pSrc, srcSize, maxSize, (RppPtr_t)pDst, &kernelSize, maskAnchor, maskLoc, noOfImages, rBorderType, handle);
    hipDeviceSynchronize();

    rppDestroyGPU(handle);
    free(srcSize);

    return(hipRppStatusTocudaNppStatus(status));
}

RppStatus
rppi_box_filter_u8_pln3_batchPD_gpu(RppPtr_t srcPtr,
                                    RppiSize *srcSize,
                                    RppiSize maxSrcSize,
                                    RppPtr_t dstPtr,
                                    Rpp32u *kernelSize,
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
    copy_param_uint(kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        box_filter_cl_batch(static_cast<cl_mem>(srcPtr),
                            static_cast<cl_mem>(dstPtr),
                            rpp::deref(rppHandle),
                            RPPI_CHN_PLANAR,
                            3);
    }
#elif defined(HIP_COMPILE)
    {
        box_filter_hip_batch(static_cast<Rpp8u*>(srcPtr),
                             static_cast<Rpp8u*>(dstPtr),
                             rpp::deref(rppHandle),
                             RPPI_CHN_PLANAR,
                             3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_box_filter_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr,
                                    RppiSize *srcSize,
                                    RppiSize maxSrcSize,
                                    RppPtr_t dstPtr,
                                    Rpp32u *kernelSize,
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
    copy_param_uint(kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        box_filter_cl_batch(static_cast<cl_mem>(srcPtr),
                            static_cast<cl_mem>(dstPtr),
                            rpp::deref(rppHandle),
                            RPPI_CHN_PACKED,
                            3);
    }
#elif defined(HIP_COMPILE)
    {
        box_filter_hip_batch(static_cast<Rpp8u*>(srcPtr),
                             static_cast<Rpp8u*>(dstPtr),
                             rpp::deref(rppHandle),
                             RPPI_CHN_PACKED,
                             3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

/******************** sobel_filter ********************/

RppStatus
rppi_sobel_filter_u8_pln1_batchPD_gpu(RppPtr_t srcPtr,
                                      RppiSize *srcSize,
                                      RppiSize maxSrcSize,
                                      RppPtr_t dstPtr,
                                      Rpp32u *sobelType,
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
    copy_param_uint(sobelType, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        sobel_filter_cl_batch(static_cast<cl_mem>(srcPtr),
                              static_cast<cl_mem>(dstPtr),
                              rpp::deref(rppHandle),
                              RPPI_CHN_PLANAR,
                              1);
    }
#elif defined(HIP_COMPILE)
    {
        sobel_filter_hip_batch(static_cast<Rpp8u*>(srcPtr),
                               static_cast<Rpp8u*>(dstPtr),
                               rpp::deref(rppHandle),
                               RPPI_CHN_PLANAR,
                               1);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_sobel_filter_u8_pln3_batchPD_gpu(RppPtr_t srcPtr,
                                      RppiSize *srcSize,
                                      RppiSize maxSrcSize,
                                      RppPtr_t dstPtr,
                                      Rpp32u *sobelType,
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
    copy_param_uint(sobelType, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        sobel_filter_cl_batch(static_cast<cl_mem>(srcPtr),
                              static_cast<cl_mem>(dstPtr),
                              rpp::deref(rppHandle),
                              RPPI_CHN_PLANAR,
                              3);
    }
#elif defined(HIP_COMPILE)
    {
        sobel_filter_hip_batch(static_cast<Rpp8u*>(srcPtr),
                               static_cast<Rpp8u*>(dstPtr),
                               rpp::deref(rppHandle),
                               RPPI_CHN_PLANAR,
                               3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_sobel_filter_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr,
                                      RppiSize *srcSize,
                                      RppiSize maxSrcSize,
                                      RppPtr_t dstPtr,
                                      Rpp32u *sobelType,
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
    copy_param_uint(sobelType, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        sobel_filter_cl_batch(static_cast<cl_mem>(srcPtr),
                              static_cast<cl_mem>(dstPtr),
                              rpp::deref(rppHandle),
                              RPPI_CHN_PACKED,
                              3);
    }
#elif defined(HIP_COMPILE)
    {
        sobel_filter_hip_batch(static_cast<Rpp8u*>(srcPtr),
                               static_cast<Rpp8u*>(dstPtr),
                               rpp::deref(rppHandle),
                               RPPI_CHN_PACKED,
                               3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

/******************** median_filter ********************/

RppStatus
rppi_median_filter_u8_pln1_batchPD_gpu(RppPtr_t srcPtr,
                                       RppiSize *srcSize,
                                       RppiSize maxSrcSize,
                                       RppPtr_t dstPtr,
                                       Rpp32u *kernelSize,
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
    copy_param_uint(kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        median_filter_cl_batch(static_cast<cl_mem>(srcPtr),
                               static_cast<cl_mem>(dstPtr),
                               rpp::deref(rppHandle),
                               RPPI_CHN_PLANAR,
                               1);
    }
#elif defined(HIP_COMPILE)
    {
        median_filter_hip_batch(static_cast<Rpp8u*>(srcPtr),
                                static_cast<Rpp8u*>(dstPtr),
                                rpp::deref(rppHandle),
                                RPPI_CHN_PLANAR,
                                1);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

NppStatus nppiFilterMedian_8u_C3R(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep, 
				NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor, Npp8u *pBuffer)
{
        int noOfImages = 1;
        int ip_channel = 3;//pkd_3
        RppiSize *srcSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize));
        RppiSize maxSize;
        srcSize->width  = oSizeROI.width;
        srcSize->height = oSizeROI.height;
        maxSize.width  = oSizeROI.width;
        maxSize.height = oSizeROI.height;
	    Rpp32u kernelSize = oMaskSize.width;

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
	    status = rppi_median_filter_u8_pkd3_batchPD_gpu((RppPtr_t)pSrc, srcSize, maxSize, (RppPtr_t)pDst, &kernelSize, noOfImages, handle);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        free(srcSize);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiFilterMedian_8u_C1R_Ctx(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor, Npp8u *pBuffer, NppStreamContext nppStreamCtx)
{
        int noOfImages = 1;
        int ip_channel = 1;//pkd_3
        RppiSize *srcSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize));
        RppiSize maxSize;
        srcSize->width  = oSizeROI.width;
        srcSize->height = oSizeROI.height;
        maxSize.width  = oSizeROI.width;
        maxSize.height = oSizeROI.height;
        Rpp32u kernelSize = oMaskSize.width;

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        status = rppi_median_filter_u8_pln1_batchPD_gpu((RppPtr_t)pSrc, srcSize, maxSize, (RppPtr_t)pDst, &kernelSize, noOfImages, handle);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        free(srcSize);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiFilterMedian_8u_C1R(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor, Npp8u *pBuffer)
{
        int noOfImages = 1;
        int ip_channel = 1;//pkd_3
        RppiSize *srcSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize));
        RppiSize maxSize;
        srcSize->width  = oSizeROI.width;
        srcSize->height = oSizeROI.height;
        maxSize.width  = oSizeROI.width;
        maxSize.height = oSizeROI.height;
        Rpp32u kernelSize = oMaskSize.width;

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        status = rppi_median_filter_u8_pln1_batchPD_gpu((RppPtr_t)pSrc, srcSize, maxSize, (RppPtr_t)pDst, &kernelSize, noOfImages, handle);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        free(srcSize);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiFilterMedian_8u_C3R_Ctx(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor, Npp8u *pBuffer, NppStreamContext nppStreamCtx)
{
        int noOfImages = 1;
        int ip_channel = 3;//pkd_3
        RppiSize *srcSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize));
        RppiSize maxSize;
        srcSize->width  = oSizeROI.width;
        srcSize->height = oSizeROI.height;
        maxSize.width  = oSizeROI.width;
        maxSize.height = oSizeROI.height;
        Rpp32u kernelSize = oMaskSize.width;

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        status = rppi_median_filter_u8_pkd3_batchPD_gpu((RppPtr_t)pSrc, srcSize, maxSize, (RppPtr_t)pDst, &kernelSize, noOfImages, handle);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        free(srcSize);

        return(hipRppStatusTocudaNppStatus(status));
}

RppStatus
rppi_median_filter_u8_pln3_batchPD_gpu(RppPtr_t srcPtr,
                                       RppiSize *srcSize,
                                       RppiSize maxSrcSize,
                                       RppPtr_t dstPtr,
                                       Rpp32u *kernelSize,
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
    copy_param_uint(kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        median_filter_cl_batch(static_cast<cl_mem>(srcPtr),
                               static_cast<cl_mem>(dstPtr),
                               rpp::deref(rppHandle),
                               RPPI_CHN_PLANAR,
                               3);
    }
#elif defined(HIP_COMPILE)
    {
        median_filter_hip_batch(static_cast<Rpp8u*>(srcPtr),
                                static_cast<Rpp8u*>(dstPtr),
                                rpp::deref(rppHandle),
                                RPPI_CHN_PLANAR,
                                3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_median_filter_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr,
                                       RppiSize *srcSize,
                                       RppiSize maxSrcSize,
                                       RppPtr_t dstPtr,
                                       Rpp32u *kernelSize,
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
    copy_param_uint(kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        median_filter_cl_batch(static_cast<cl_mem>(srcPtr),
                               static_cast<cl_mem>(dstPtr),
                               rpp::deref(rppHandle),
                               RPPI_CHN_PACKED,
                               3);
    }
#elif defined(HIP_COMPILE)
    {
        median_filter_hip_batch(static_cast<Rpp8u*>(srcPtr),
                                static_cast<Rpp8u*>(dstPtr),
                                rpp::deref(rppHandle),
                                RPPI_CHN_PACKED,
                                3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

/******************** non_max_suppression ********************/

RppStatus
rppi_non_max_suppression_u8_pln1_batchPD_gpu(RppPtr_t srcPtr,
                                             RppiSize *srcSize,
                                             RppiSize maxSrcSize,
                                             RppPtr_t dstPtr,
                                             Rpp32u *kernelSize,
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
    copy_param_uint(kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        non_max_suppression_cl_batch(static_cast<cl_mem>(srcPtr),
                                     static_cast<cl_mem>(dstPtr),
                                     rpp::deref(rppHandle),
                                     RPPI_CHN_PLANAR,
                                     1);
    }
#elif defined(HIP_COMPILE)
    {
        non_max_suppression_hip_batch(static_cast<Rpp8u*>(srcPtr),
                                      static_cast<Rpp8u*>(dstPtr),
                                      rpp::deref(rppHandle),
                                      RPPI_CHN_PLANAR,
                                      1);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_non_max_suppression_u8_pln3_batchPD_gpu(RppPtr_t srcPtr,
                                             RppiSize *srcSize,
                                             RppiSize maxSrcSize,
                                             RppPtr_t dstPtr,
                                             Rpp32u *kernelSize,
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
    copy_param_uint(kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        non_max_suppression_cl_batch(static_cast<cl_mem>(srcPtr),
                                     static_cast<cl_mem>(dstPtr),
                                     rpp::deref(rppHandle),
                                     RPPI_CHN_PLANAR,
                                     3);
    }
#elif defined(HIP_COMPILE)
    {
        non_max_suppression_hip_batch(static_cast<Rpp8u*>(srcPtr),
                                      static_cast<Rpp8u*>(dstPtr),
                                      rpp::deref(rppHandle),
                                      RPPI_CHN_PLANAR,
                                      3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_non_max_suppression_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr,
                                             RppiSize *srcSize,
                                             RppiSize maxSrcSize,
                                             RppPtr_t dstPtr,
                                             Rpp32u *kernelSize,
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
    copy_param_uint(kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        non_max_suppression_cl_batch(static_cast<cl_mem>(srcPtr),
                                     static_cast<cl_mem>(dstPtr),
                                     rpp::deref(rppHandle),
                                     RPPI_CHN_PACKED,
                                     3);
    }
#elif defined(HIP_COMPILE)
    {
        non_max_suppression_hip_batch(static_cast<Rpp8u*>(srcPtr),
                                      static_cast<Rpp8u*>(dstPtr),
                                      rpp::deref(rppHandle),
                                      RPPI_CHN_PACKED,
                                      3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

/******************** gaussian_filter ********************/

RppStatus
rppi_gaussian_filter_u8_pln1_batchPD_gpu(RppPtr_t srcPtr,
                                         RppiSize *srcSize,
                                         RppiSize maxSrcSize,
                                         RppPtr_t dstPtr,
                                         Rpp32f *stdDev,
                                         Rpp32u *kernelSize,
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
    copy_param_float(stdDev, rpp::deref(rppHandle), paramIndex++);
    copy_param_uint(kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        gaussian_filter_cl_batch(static_cast<cl_mem>(srcPtr),
                                 static_cast<cl_mem>(dstPtr),
                                 rpp::deref(rppHandle),
                                 RPPI_CHN_PLANAR,
                                 1);
    }
#elif defined(HIP_COMPILE)
    {
        gaussian_filter_hip_batch(static_cast<Rpp8u*>(srcPtr),
                                  static_cast<Rpp8u*>(dstPtr),
                                  rpp::deref(rppHandle),
                                  RPPI_CHN_PLANAR,
                                  1);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_gaussian_filter_u8_pln3_batchPD_gpu(RppPtr_t srcPtr,
                                         RppiSize *srcSize,
                                         RppiSize maxSrcSize,
                                         RppPtr_t dstPtr,
                                         Rpp32f *stdDev,
                                         Rpp32u *kernelSize,
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
    copy_param_float(stdDev, rpp::deref(rppHandle), paramIndex++);
    copy_param_uint(kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        gaussian_filter_cl_batch(static_cast<cl_mem>(srcPtr),
                                 static_cast<cl_mem>(dstPtr),
                                 rpp::deref(rppHandle),
                                 RPPI_CHN_PLANAR,
                                 3);
    }
#elif defined(HIP_COMPILE)
    {
        gaussian_filter_hip_batch(static_cast<Rpp8u*>(srcPtr),
                                  static_cast<Rpp8u*>(dstPtr),
                                  rpp::deref(rppHandle),
                                  RPPI_CHN_PLANAR,
                                  3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_gaussian_filter_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr,
                                         RppiSize *srcSize,
                                         RppiSize maxSrcSize,
                                         RppPtr_t dstPtr,
                                         Rpp32f *stdDev,
                                         Rpp32u *kernelSize,
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
    copy_param_float(stdDev, rpp::deref(rppHandle), paramIndex++);
    copy_param_uint(kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        gaussian_filter_cl_batch(static_cast<cl_mem>(srcPtr),
                                 static_cast<cl_mem>(dstPtr),
                                 rpp::deref(rppHandle),
                                 RPPI_CHN_PACKED,
                                 3);
    }
#elif defined(HIP_COMPILE)
    {
        gaussian_filter_hip_batch(static_cast<Rpp8u*>(srcPtr),
                                  static_cast<Rpp8u*>(dstPtr),
                                  rpp::deref(rppHandle),
                                  RPPI_CHN_PACKED,
                                  3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

/******************** nonlinear_filter ********************/

RppStatus
rppi_nonlinear_filter_u8_pln1_batchPD_gpu(RppPtr_t srcPtr,
                                          RppiSize *srcSize,
                                          RppiSize maxSrcSize,
                                          RppPtr_t dstPtr,
                                          Rpp32u *kernelSize,
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
    copy_param_uint(kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        median_filter_cl_batch(static_cast<cl_mem>(srcPtr),
                               static_cast<cl_mem>(dstPtr),
                               rpp::deref(rppHandle),
                               RPPI_CHN_PLANAR,
                               1);
    }
#elif defined(HIP_COMPILE)
    {
        median_filter_hip_batch(static_cast<Rpp8u*>(srcPtr),
                                static_cast<Rpp8u*>(dstPtr),
                                rpp::deref(rppHandle),
                                RPPI_CHN_PLANAR,
                                1);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_nonlinear_filter_u8_pln3_batchPD_gpu(RppPtr_t srcPtr,
                                          RppiSize *srcSize,
                                          RppiSize maxSrcSize,
                                          RppPtr_t dstPtr,
                                          Rpp32u *kernelSize,
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
    copy_param_uint(kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        median_filter_cl_batch(static_cast<cl_mem>(srcPtr),
                               static_cast<cl_mem>(dstPtr),
                               rpp::deref(rppHandle),
                               RPPI_CHN_PLANAR,
                               3);
    }
#elif defined(HIP_COMPILE)
    {
        median_filter_hip_batch(static_cast<Rpp8u*>(srcPtr),
                                static_cast<Rpp8u*>(dstPtr),
                                rpp::deref(rppHandle),
                                RPPI_CHN_PLANAR,
                                3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_nonlinear_filter_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr,
                                          RppiSize *srcSize,
                                          RppiSize maxSrcSize,
                                          RppPtr_t dstPtr,
                                          Rpp32u *kernelSize,
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
    copy_param_uint(kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        median_filter_cl_batch(static_cast<cl_mem>(srcPtr),
                               static_cast<cl_mem>(dstPtr),
                               rpp::deref(rppHandle),
                               RPPI_CHN_PACKED,
                               3);
    }
#elif defined(HIP_COMPILE)
    {
        median_filter_hip_batch(static_cast<Rpp8u*>(srcPtr),
                                static_cast<Rpp8u*>(dstPtr),
                                rpp::deref(rppHandle),
                                RPPI_CHN_PACKED,
                                3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

/******************** custom_convolution ********************/

RppStatus
rppi_custom_convolution_u8_pln1_batchPD_gpu(RppPtr_t srcPtr,
                                            RppiSize *srcSize,
                                            RppiSize maxSrcSize,
                                            RppPtr_t dstPtr,
                                            RppPtr_t kernel,
                                            RppiSize *kernelSize,
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
        custom_convolution_cl_batch(static_cast<cl_mem>(srcPtr),
                                    static_cast<cl_mem>(dstPtr),
                                    static_cast<Rpp32f*>(kernel),
                                    kernelSize[0],
                                    rpp::deref(rppHandle),
                                    RPPI_CHN_PLANAR,
                                    1);
    }
#elif defined(HIP_COMPILE)
    {
        custom_convolution_hip_batch(static_cast<Rpp8u*>(srcPtr),
                                     static_cast<Rpp8u*>(dstPtr),
                                     static_cast<Rpp32f*>(kernel),
                                     kernelSize[0],
                                     rpp::deref(rppHandle),
                                     RPPI_CHN_PLANAR,
                                     1);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_custom_convolution_u8_pln3_batchPD_gpu(RppPtr_t srcPtr,
                                            RppiSize *srcSize,
                                            RppiSize maxSrcSize,
                                            RppPtr_t dstPtr,
                                            RppPtr_t kernel,
                                            RppiSize *kernelSize,
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
        custom_convolution_cl_batch(static_cast<cl_mem>(srcPtr),
                                    static_cast<cl_mem>(dstPtr),
                                    static_cast<Rpp32f*>(kernel),
                                    kernelSize[0],
                                    rpp::deref(rppHandle),
                                    RPPI_CHN_PLANAR,
                                    3);
    }
#elif defined(HIP_COMPILE)
    {
        custom_convolution_hip_batch(static_cast<Rpp8u*>(srcPtr),
                                     static_cast<Rpp8u*>(dstPtr),
                                     static_cast<Rpp32f*>(kernel),
                                     kernelSize[0],
                                     rpp::deref(rppHandle),
                                     RPPI_CHN_PLANAR,
                                     3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_custom_convolution_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr,
                                            RppiSize *srcSize,
                                            RppiSize maxSrcSize,
                                            RppPtr_t dstPtr,
                                            RppPtr_t kernel,
                                            RppiSize *kernelSize,
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
        custom_convolution_cl_batch(static_cast<cl_mem>(srcPtr),
                                    static_cast<cl_mem>(dstPtr),
                                    static_cast<Rpp32f*>(kernel),
                                    kernelSize[0],
                                    rpp::deref(rppHandle),
                                    RPPI_CHN_PACKED,
                                    3);
    }
#elif defined(HIP_COMPILE)
    {
        custom_convolution_hip_batch(static_cast<Rpp8u*>(srcPtr),
                                     static_cast<Rpp8u*>(dstPtr),
                                     static_cast<Rpp32f*>(kernel),
                                     kernelSize[0],
                                     rpp::deref(rppHandle),
                                     RPPI_CHN_PACKED,
                                     3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

NppStatus nppiGradientVectorPrewittBorder_8u16s_C1R(const Npp8u *pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s *pDstX, int nDstXStep, Npp16s *pDstY, 
						int nDstYStep, Npp16s *pDstMag, int nDstMagStep, Npp32f *pDstAngle, int nDstAngleStep, NppiSize oSizeROI, NppiMaskSize eMaskSize, 
						NppiNorm eNorm, NppiBorderType eBorderType)
{
    RppiNorm rNorm;
    if(eNorm == nppiNormL1){
        rNorm = rppiNormL1;
    } else if(eNorm == nppiNormL2){ 
		rNorm = rppiNormL2;
	}
	else {
        return(hipRppStatusTocudaNppStatus(RPP_ERROR_INVALID_ARGUMENTS));
    }
    
    int noOfImages = 1;
    int ip_channel = 1;//pln1
    RppiSize *srcSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize));
    RppiSize maxSize;
	RppiPoint maskLoc;
	RppiBorderType rBorderType;
    srcSize->width  = oSizeROI.width;
    srcSize->height = oSizeROI.height;
    maxSize.width  = oSrcSize.width;
    maxSize.height = oSrcSize.height;
	maskLoc.x = oSrcOffset.x;
	maskLoc.y = oSrcOffset.y;
		
	if(eBorderType == NPP_BORDER_UNDEFINED || eBorderType == NPP_BORDER_NONE){
        rBorderType = RPP_BORDER_UNDEFINED;
    } else if(eBorderType == NPP_BORDER_CONSTANT) {
        rBorderType = RPP_BORDER_CONSTANT;
    } else if(eBorderType == NPP_BORDER_REPLICATE) {
        rBorderType = RPP_BORDER_REPLICATE;
    } else if(eBorderType == NPP_BORDER_WRAP) {
        rBorderType = RPP_BORDER_WRAP;
    } else if(eBorderType == NPP_BORDER_MIRROR) {
        rBorderType = RPP_BORDER_MIRROR;
    }

    rppHandle_t handle;
    hipStream_t stream;
    hipStreamCreate(&stream);
    rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);
       
    RppStatus status;
    status = nppi_prewitt_filter_u8_pln1_batchPD_gpu((RppPtr_t)pSrc, srcSize, maxSize, (RppPtr_t)pDstX, (RppPtr_t)pDstY, (RppPtr_t)pDstMag, (RppPtr_t)pDstAngle, maskLoc, rNorm, noOfImages, rBorderType, handle);
    hipDeviceSynchronize();

    rppDestroyGPU(handle);
    free(srcSize);

    return(hipRppStatusTocudaNppStatus(status));
}

RppStatus
nppi_prewitt_filter_u8_pln1_batchPD_gpu(RppPtr_t srcPtr,
                                        RppiSize *srcSize,
                                        RppiSize maxSrcSize,
                                        RppPtr_t dstPtrx,
					                    RppPtr_t dstPtry,
					                    RppPtr_t pDstMag,
					                    RppPtr_t pDstAngle,
					                    RppiPoint maskLoc,
					                    RppiNorm eNorm,
                                        Rpp32u nbatchSize,
					                    RppiBorderType rBorderType,
                                        rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    roiPoints.x = maskLoc.x;
    roiPoints.y = maskLoc.y;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    Rpp32u paramIndex = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    copy_roi(roiPoints, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
    //copy_param_uint(kernelSize, rpp::deref(rppHandle), paramIndex++);

    prewitt_filter_npp_batch(static_cast<Rpp8u*>(srcPtr),
                             static_cast<Rpp16s*>(dstPtrx),
			                 static_cast<Rpp16s*>(dstPtry),
			                 static_cast<Rpp16s*>(pDstMag),
			                 static_cast<Rpp32f*>(pDstAngle),
                             rpp::deref(rppHandle),
                             RPPI_CHN_PLANAR,
                             1,
			                 eNorm,
			                 rBorderType);
	
    return RPP_SUCCESS;
}

NppStatus nppiLabelMarkersUFGetBufferSize_32u_C1R(NppiSize oSizeROI, int *hpBufferSize)
{   
    if (hpBufferSize == nullptr) {  
        return NPP_NULL_POINTER_ERROR;  
    }  
  
    int width = oSizeROI.width;  
    int height = oSizeROI.height;  
 
    if (width <= 0 || height <= 0) {  
        return NPP_ERROR;  
    }  

    int tempBufferSize = width * height * sizeof(unsigned int);   
 
    *hpBufferSize = tempBufferSize;  

    return NPP_SUCCESS;  
} 

NppStatus nppiCompressMarkerLabelsGetBufferSize_32u_C1R(int nStartingNumber, int *hpBufferSize)
{
    if (hpBufferSize == NULL || nStartingNumber < 0) {  
        return NPP_NULL_POINTER_ERROR;  
    }  
    
    int requiredSize = nStartingNumber * sizeof(int);  
     
    *hpBufferSize = requiredSize;  

    return NPP_SUCCESS; 
}

NppStatus nppiLabelMarkersUF_8u32u_C1R_Ctx(Npp8u *pSrc, int nSrcStep, Npp32u *pDst, int nDstStep, NppiSize oSizeROI, NppiNorm eNorm, Npp8u *pBuffer, NppStreamContext nppStreamCtx)
{
        RppiNorm rNorm;
        if(eNorm == nppiNormL1){
                rNorm = rppiNormL1;
        } else if(eNorm == nppiNormInf){
                rNorm = rppiNormInf;
        } else {
                return(hipRppStatusTocudaNppStatus(RPP_ERROR_INVALID_ARGUMENTS));
        }
        
        int noOfImages = 1;
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
        status = rppi_labelmarkers_u8_pln1_batchPD_gpu((RppPtr_t)pSrc, srcSize, maxSize, (RppPtr_t)pDst, rNorm, noOfImages, handle);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        free(srcSize);

        return(hipRppStatusTocudaNppStatus(status));
}

RppStatus
rppi_labelmarkers_u8_pln1_batchPD_gpu(RppPtr_t srcPtr,
                                    RppiSize *srcSize,
                                    RppiSize maxSrcSize,
                                    RppPtr_t dstPtr,
                                    RppiNorm eNorm,
                                    Rpp32u nbatchSize,
                                    rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    //roiPoints.x = (maxSrcSize.width - srcSize->width) / 2;;
    //roiPoints.y = (maxSrcSize.height - srcSize->height) / 2;
    //roiPoints.roiHeight = (maxSrcSize.height + srcSize->height) / 2;
    //roiPoints.roiWidth = (maxSrcSize.width + srcSize->width) / 2;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    copy_roi(roiPoints, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

    labelmarkers_npp_batch(static_cast<Rpp8u*>(srcPtr),
                           static_cast<Rpp32u*>(dstPtr),
                           rpp::deref(rppHandle),
                           RPPI_CHN_PLANAR,
                           eNorm);

    return RPP_SUCCESS;
}

NppStatus nppiLabelMarkersUFBatch_8u32u_C1R_Advanced_Ctx(const NppiImageDescriptor *pSrcBatchList, NppiImageDescriptor *pDstBatchList, int nBatchSize, NppiSize oMaxSizeROI, 
							NppiNorm eNorm, NppStreamContext nppStreamCtx)
{
    RppiNorm rNorm;
    if(eNorm == nppiNormInf){
        rNorm = rppiNormInf;
    } else if(eNorm == nppiNormL1){
        rNorm = rppiNormL1;
    } else {
        return(hipRppStatusTocudaNppStatus(RPP_ERROR_INVALID_ARGUMENTS));
    }
    
    int noOfImages = 1;
    RppiSize *srcSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize));
    RppiSize maxSize;
    //srcSize->width  = oMaxSizeROI.width;
    //srcSize->height = oMaxSizeROI.height;
    maxSize.width  = oMaxSizeROI.width;
    maxSize.height = oMaxSizeROI.height;

    rppHandle_t handle;
    hipStream_t stream;
    hipStreamCreate(&stream);
    rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

    RppStatus status;
		
	for (int batchIndex = 0; batchIndex < nBatchSize; ++batchIndex) {  
		const void* src = pSrcBatchList[batchIndex].pData;  
		void* dst = pDstBatchList[batchIndex].pData;  
		srcSize->width = pSrcBatchList[batchIndex].oSize.width;
		srcSize->height = pSrcBatchList[batchIndex].oSize.height;
		//maxSize.width  = pSrcBatchList[batchIndex].oSize.width;
		//maxSize.height = pSrcBatchList[batchIndex].oSize.height;
		status = rppi_labelmarkers_u8_pln1_batchPD_gpu((RppPtr_t)src, srcSize, maxSize, (RppPtr_t)dst, rNorm, noOfImages, handle);
		hipDeviceSynchronize();
		if(status != 0){
            free(srcSize);
			return(hipRppStatusTocudaNppStatus(status));
		}
	}
		
    rppDestroyGPU(handle);
    free(srcSize);

    return NPP_SUCCESS;
}

NppStatus nppiCompressMarkerLabelsUF_32u_C1IR(Npp32u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nStartingNumber, int *pNewNumber, Npp8u *pBuffer)
{
    int noOfImages = 1;
    RppiSize *srcSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize));
    RppiSize maxSize;
    srcSize->width  = oSizeROI.width;
    srcSize->height = oSizeROI.height;
    maxSize.width  = oSizeROI.width;
    maxSize.height = oSizeROI.height;
 	int *hNewNumber;
	hipMalloc((void**)&hNewNumber, sizeof(int)); 

    rppHandle_t handle;
    hipStream_t stream;
    hipStreamCreate(&stream);
    rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

    RppStatus status;
    status = rppi_compressmarkers_u8_pln1_gpu((RppPtr_t)pSrcDst, srcSize, maxSize, hNewNumber, nStartingNumber, noOfImages, handle);
	hipMemcpy(pNewNumber, hNewNumber, sizeof(int), hipMemcpyDeviceToHost);
    hipDeviceSynchronize();

    rppDestroyGPU(handle);
    free(srcSize);
	hipFree(hNewNumber);

    return(hipRppStatusTocudaNppStatus(status));
}

RppStatus
rppi_compressmarkers_u8_pln1_gpu(RppPtr_t srcPtr,
                                    RppiSize *srcSize,
                                    RppiSize maxSrcSize,
                                    Rpp32s *pNewNumber,
                                    Rpp32s nStartingNumber,
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

    compressmarkers_npp(static_cast<Rpp32u*>(srcPtr),
                              rpp::deref(rppHandle),
                              RPPI_CHN_PLANAR,
                              pNewNumber,
                              nStartingNumber);

    return RPP_SUCCESS;
}

NppStatus nppiCompressMarkerLabelsUFBatch_32u_C1IR_Advanced_Ctx(NppiImageDescriptor *pSrcDstBatchList, NppiBufferDescriptor *pBufferList, unsigned int *pNewMaxLabelIDList,
int nBatchSize, NppiSize oMaxSizeROI,int nLargestPerImageBufferSize, NppStreamContext nppStreamCtx)														
{
	printf("nppiCompressMarkerLabelsUFBatch_32u_C1IR_Advanced_Ctx not support.\n");
    return NPP_SUCCESS;
}
RppStatus
rppi_compressmarkers_u8_pln1_batchPD_gpu(RppPtr_t srcPtr,
                                        RppiSize *srcSize,
                                        RppiSize maxSrcSize,
				                        RppBufferDescriptor *pBufferBatch,
				                        Rpp32u *pNewMaxLabelID,
				                        Rpp32s nPerImageBufferSize,
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

    compressmarkers_npp_batch(static_cast<Rpp32u*>(srcPtr),
                              rpp::deref(rppHandle),
                              RPPI_CHN_PLANAR,
                              pBufferBatch,
			                  pNewMaxLabelID,
			                  nPerImageBufferSize);

    return RPP_SUCCESS;
}

#endif // GPU_SUPPORT

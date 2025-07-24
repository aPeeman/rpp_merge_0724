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
#include "rppi_geometry_transforms.h"
#include "cpu/host_geometry_transforms.hpp"

#ifdef HIP_COMPILE
#include "rpp_hip_common.hpp"
#include "hip/hip_declarations.hpp"
#elif defined(OCL_COMPILE)
#include "rpp_cl_common.hpp"
#include "cl/cl_declarations.hpp"
#endif //backend

/******************** flip ********************/

RppStatus
rppi_flip_u8_pln1_batchPD_host(RppPtr_t srcPtr,
                               RppiSize *srcSize,
                               RppiSize maxSrcSize,
                               RppPtr_t dstPtr,
                               Rpp32u *flipAxis,
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

    flip_host_batch<Rpp8u>(static_cast<Rpp8u *>(srcPtr),
                           srcSize,
                           rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                           static_cast<Rpp8u *>(dstPtr),
                           flipAxis,
                           rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                           rpp::deref(rppHandle).GetBatchSize(),
                           RPPI_CHN_PLANAR,
                           1,
                           rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_flip_u8_pln3_batchPD_host(RppPtr_t srcPtr,
                               RppiSize *srcSize,
                               RppiSize maxSrcSize,
                               RppPtr_t dstPtr,
                               Rpp32u *flipAxis,
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

    flip_host_batch<Rpp8u>(static_cast<Rpp8u *>(srcPtr),
                           srcSize,
                           rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                           static_cast<Rpp8u *>(dstPtr),
                           flipAxis,
                           rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                           rpp::deref(rppHandle).GetBatchSize(),
                           RPPI_CHN_PLANAR,
                           3,
                           rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_flip_u8_pkd3_batchPD_host(RppPtr_t srcPtr,
                               RppiSize *srcSize,
                               RppiSize maxSrcSize,
                               RppPtr_t dstPtr,
                               Rpp32u *flipAxis,
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

    flip_host_batch<Rpp8u>(static_cast<Rpp8u *>(srcPtr),
                           srcSize,
                           rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                           static_cast<Rpp8u *>(dstPtr),
                           flipAxis,
                           rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                           rpp::deref(rppHandle).GetBatchSize(),
                           RPPI_CHN_PACKED,
                           3,
                           rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

/******************** resize ********************/

RppStatus resize_host_helper(RppiChnFormat chn_format,
                             Rpp32u num_of_channels,
                             RPPTensorDataType tensorInType,
                             RPPTensorDataType tensorOutType,
                             RppPtr_t srcPtr,
                             RppiSize *srcSize,
                             RppiSize maxSrcSize,
                             RppPtr_t dstPtr,
                             RppiSize *dstSize,
                             RppiSize maxDstSize,
                             Rpp32u outputFormatToggle,
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
    copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));

    if (tensorInType == RPPTensorDataType::U8)
    {
        if (tensorOutType == RPPTensorDataType::U8)
        {
            resize_host_batch<Rpp8u, Rpp8u>(static_cast<Rpp8u *>(srcPtr),
                                            srcSize,
                                            rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                            static_cast<Rpp8u *>(dstPtr),
                                            dstSize,
                                            rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
                                            rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                                            outputFormatToggle,
                                            rpp::deref(rppHandle).GetBatchSize(),
                                            chn_format,
                                            num_of_channels,
                                            rpp::deref(rppHandle));
        }
        else if (tensorOutType == RPPTensorDataType::FP16)
        {
            resize_host_batch<Rpp8u, Rpp16f>(static_cast<Rpp8u *>(srcPtr),
                                             srcSize,
                                             rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                             static_cast<Rpp16f *>(dstPtr),
                                             dstSize,
                                             rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
                                             rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                                             outputFormatToggle,
                                             rpp::deref(rppHandle).GetBatchSize(),
                                             chn_format,
                                             num_of_channels,
                                             rpp::deref(rppHandle));
        }
        else if (tensorOutType == RPPTensorDataType::FP32)
        {
            resize_host_batch<Rpp8u, Rpp32f>(static_cast<Rpp8u *>(srcPtr),
                                             srcSize,
                                             rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                             static_cast<Rpp32f *>(dstPtr),
                                             dstSize,
                                             rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
                                             rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                                             outputFormatToggle,
                                             rpp::deref(rppHandle).GetBatchSize(),
                                             chn_format,
                                             num_of_channels,
                                             rpp::deref(rppHandle));
        }
        else if (tensorOutType == RPPTensorDataType::I8)
        {
            resize_u8_i8_host_batch<Rpp8u, Rpp8s>(static_cast<Rpp8u *>(srcPtr),
                                                  srcSize,
                                                  rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                                  static_cast<Rpp8s *>(dstPtr),
                                                  dstSize,
                                                  rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
                                                  rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                                                  outputFormatToggle,
                                                  rpp::deref(rppHandle).GetBatchSize(),
                                                  chn_format,
                                                  num_of_channels,
                                                  rpp::deref(rppHandle));
        }
    }
    else if (tensorInType == RPPTensorDataType::FP16)
    {
        resize_host_batch<Rpp16f, Rpp16f>(static_cast<Rpp16f *>(srcPtr),
                                          srcSize,
                                          rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                          static_cast<Rpp16f *>(dstPtr),
                                          dstSize,
                                          rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
                                          rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                                          outputFormatToggle,
                                          rpp::deref(rppHandle).GetBatchSize(),
                                          chn_format,
                                          num_of_channels,
                                          rpp::deref(rppHandle));
    }
    else if (tensorInType == RPPTensorDataType::FP32)
    {
        resize_host_batch<Rpp32f, Rpp32f>(static_cast<Rpp32f *>(srcPtr),
                                          srcSize,
                                          rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                          static_cast<Rpp32f *>(dstPtr),
                                          dstSize,
                                          rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
                                          rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                                          outputFormatToggle,
                                          rpp::deref(rppHandle).GetBatchSize(),
                                          chn_format,
                                          num_of_channels,
                                          rpp::deref(rppHandle));
    }
    else if (tensorInType == RPPTensorDataType::I8)
    {
        resize_host_batch<Rpp8s, Rpp8s>(static_cast<Rpp8s *>(srcPtr),
                                        srcSize,
                                        rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                        static_cast<Rpp8s *>(dstPtr),
                                        dstSize,
                                        rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
                                        rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                                        outputFormatToggle,
                                        rpp::deref(rppHandle).GetBatchSize(),
                                        chn_format,
                                        num_of_channels,
                                        rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
}

RppStatus
rppi_resize_u8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_host_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::U8, RPPTensorDataType::U8, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_f16_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_host_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::FP16, RPPTensorDataType::FP16, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_f32_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_host_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::FP32, RPPTensorDataType::FP32, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_i8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_host_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::I8, RPPTensorDataType::I8, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_u8_f16_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_host_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::U8, RPPTensorDataType::FP16, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_u8_f32_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_host_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::U8, RPPTensorDataType::FP32, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_u8_i8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_host_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::U8, RPPTensorDataType::I8, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_u8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_host_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::U8, RPPTensorDataType::U8, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_f16_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_host_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::FP16, RPPTensorDataType::FP16, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_f32_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_host_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::FP32, RPPTensorDataType::FP32, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_i8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_host_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::I8, RPPTensorDataType::I8, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_u8_f16_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_host_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::U8, RPPTensorDataType::FP16, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_u8_f32_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_host_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::U8, RPPTensorDataType::FP32, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_u8_i8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_host_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::U8, RPPTensorDataType::I8, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_u8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::U8, RPPTensorDataType::U8, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_f16_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::FP16, RPPTensorDataType::FP16, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_f32_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::FP32, RPPTensorDataType::FP32, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_i8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::I8, RPPTensorDataType::I8, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_u8_f16_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::U8, RPPTensorDataType::FP16, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_u8_f32_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::U8, RPPTensorDataType::FP32, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_u8_i8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::U8, RPPTensorDataType::I8, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, outputFormatToggle, nbatchSize, rppHandle));
}

/******************** resize_crop ********************/

RppStatus resize_crop_host_helper(RppiChnFormat chn_format,
                                  Rpp32u num_of_channels,
                                  RPPTensorDataType tensor_type,
                                  RppPtr_t srcPtr,
                                  RppiSize *srcSize,
                                  RppiSize maxSrcSize,
                                  RppPtr_t dstPtr,
                                  RppiSize *dstSize,
                                  RppiSize maxDstSize,
                                  Rpp32u *xRoiBegin,
                                  Rpp32u *xRoiEnd,
                                  Rpp32u *yRoiBegin,
                                  Rpp32u *yRoiEnd,
                                  Rpp32u outputFormatToggle,
                                  Rpp32u nbatchSize,
                                  rppHandle_t rppHandle)
{
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
    copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));

    if (tensor_type == RPPTensorDataType::U8)
    {
        resize_crop_host_batch<Rpp8u>(static_cast<Rpp8u *>(srcPtr),
                                      srcSize,
                                      rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                      static_cast<Rpp8u *>(dstPtr),
                                      dstSize,
                                      rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
                                      xRoiBegin,
                                      xRoiEnd,
                                      yRoiBegin,
                                      yRoiEnd,
                                      outputFormatToggle,
                                      rpp::deref(rppHandle).GetBatchSize(),
                                      chn_format,
                                      num_of_channels,
                                      rpp::deref(rppHandle));
    }
    else if (tensor_type == RPPTensorDataType::FP16)
    {
        resize_crop_host_batch<Rpp16f>(static_cast<Rpp16f *>(srcPtr),
                                       srcSize,
                                       rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                       static_cast<Rpp16f *>(dstPtr),
                                       dstSize,
                                       rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
                                       xRoiBegin,
                                       xRoiEnd,
                                       yRoiBegin,
                                       yRoiEnd,
                                       outputFormatToggle,
                                       rpp::deref(rppHandle).GetBatchSize(),
                                       chn_format,
                                       num_of_channels,
                                       rpp::deref(rppHandle));
    }
    else if (tensor_type == RPPTensorDataType::FP32)
    {
        resize_crop_host_batch<Rpp32f>(static_cast<Rpp32f *>(srcPtr),
                                       srcSize,
                                       rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                       static_cast<Rpp32f *>(dstPtr),
                                       dstSize,
                                       rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
                                       xRoiBegin,
                                       xRoiEnd,
                                       yRoiBegin,
                                       yRoiEnd,
                                       outputFormatToggle,
                                       rpp::deref(rppHandle).GetBatchSize(),
                                       chn_format,
                                       num_of_channels,
                                       rpp::deref(rppHandle));
    }
    else if (tensor_type == RPPTensorDataType::I8)
    {
        resize_crop_host_batch<Rpp8s>(static_cast<Rpp8s *>(srcPtr),
                                      srcSize,
                                      rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                      static_cast<Rpp8s *>(dstPtr),
                                      dstSize,
                                      rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
                                      xRoiBegin,
                                      xRoiEnd,
                                      yRoiBegin,
                                      yRoiEnd,
                                      outputFormatToggle,
                                      rpp::deref(rppHandle).GetBatchSize(),
                                      chn_format,
                                      num_of_channels,
                                      rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
}

RppStatus
rppi_resize_crop_u8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_crop_host_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::U8, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, xRoiBegin, xRoiEnd, yRoiBegin, yRoiEnd, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_crop_f16_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_crop_host_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::FP16, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, xRoiBegin, xRoiEnd, yRoiBegin, yRoiEnd, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_crop_f32_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_crop_host_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::FP32, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, xRoiBegin, xRoiEnd, yRoiBegin, yRoiEnd, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_crop_i8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_crop_host_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::I8, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, xRoiBegin, xRoiEnd, yRoiBegin, yRoiEnd, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_crop_u8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_crop_host_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::U8, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, xRoiBegin, xRoiEnd, yRoiBegin, yRoiEnd, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_crop_f16_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_crop_host_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::FP16, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, xRoiBegin, xRoiEnd, yRoiBegin, yRoiEnd, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_crop_f32_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_crop_host_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::FP32, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, xRoiBegin, xRoiEnd, yRoiBegin, yRoiEnd, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_crop_i8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_crop_host_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::I8, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, xRoiBegin, xRoiEnd, yRoiBegin, yRoiEnd, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_crop_u8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_crop_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::U8, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, xRoiBegin, xRoiEnd, yRoiBegin, yRoiEnd, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_crop_f16_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_crop_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::FP16, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, xRoiBegin, xRoiEnd, yRoiBegin, yRoiEnd, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_crop_f32_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_crop_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::FP32, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, xRoiBegin, xRoiEnd, yRoiBegin, yRoiEnd, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_crop_i8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_crop_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::I8, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, xRoiBegin, xRoiEnd, yRoiBegin, yRoiEnd, outputFormatToggle, nbatchSize, rppHandle));
}

/******************** rotate ********************/

RppStatus rotate_host_helper(RppiChnFormat chn_format,
                             Rpp32u num_of_channels,
                             RPPTensorDataType tensor_type,
                             RppPtr_t srcPtr,
                             RppiSize *srcSize,
                             RppiSize maxSrcSize,
                             RppPtr_t dstPtr,
                             RppiSize *dstSize,
                             RppiSize maxDstSize,
                             Rpp32f *angleDeg,
                             Rpp32u outputFormatToggle,
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
    copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));

    if (tensor_type == RPPTensorDataType::U8)
    {
        rotate_host_batch<Rpp8u>(static_cast<Rpp8u *>(srcPtr),
                                 srcSize,
                                 rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                 static_cast<Rpp8u *>(dstPtr),
                                 dstSize,
                                 rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
                                 angleDeg,
                                 rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                                 outputFormatToggle,
                                 rpp::deref(rppHandle).GetBatchSize(),
                                 chn_format,
                                 num_of_channels,
                                 rpp::deref(rppHandle));
    }
    else if (tensor_type == RPPTensorDataType::FP16)
    {
        rotate_host_batch<Rpp16f>(static_cast<Rpp16f *>(srcPtr),
                                  srcSize,
                                  rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                  static_cast<Rpp16f *>(dstPtr),
                                  dstSize,
                                  rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
                                  angleDeg,
                                  rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                                  outputFormatToggle,
                                  rpp::deref(rppHandle).GetBatchSize(),
                                  chn_format,
                                  num_of_channels,
                                  rpp::deref(rppHandle));
    }
    else if (tensor_type == RPPTensorDataType::FP32)
    {
        rotate_host_batch<Rpp32f>(static_cast<Rpp32f *>(srcPtr),
                                  srcSize,
                                  rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                  static_cast<Rpp32f *>(dstPtr),
                                  dstSize,
                                  rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
                                  angleDeg,
                                  rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                                  outputFormatToggle,
                                  rpp::deref(rppHandle).GetBatchSize(),
                                  chn_format,
                                  num_of_channels,
                                  rpp::deref(rppHandle));
    }
    else if (tensor_type == RPPTensorDataType::I8)
    {
        rotate_host_batch<Rpp8s>(static_cast<Rpp8s *>(srcPtr),
                                 srcSize,
                                 rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                 static_cast<Rpp8s *>(dstPtr),
                                 dstSize,
                                 rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
                                 angleDeg,
                                 rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                                 outputFormatToggle,
                                 rpp::deref(rppHandle).GetBatchSize(),
                                 chn_format,
                                 num_of_channels,
                                 rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
}

RppStatus
rppi_rotate_u8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *angleDeg, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (rotate_host_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::U8, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, angleDeg, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_rotate_f16_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *angleDeg, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (rotate_host_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::FP16, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, angleDeg, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_rotate_f32_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *angleDeg, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (rotate_host_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::FP32, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, angleDeg, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_rotate_i8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *angleDeg, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (rotate_host_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::I8, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, angleDeg, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_rotate_u8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *angleDeg, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (rotate_host_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::U8, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, angleDeg, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_rotate_f16_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *angleDeg, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (rotate_host_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::FP16, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, angleDeg, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_rotate_f32_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *angleDeg, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (rotate_host_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::FP32, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, angleDeg, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_rotate_i8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *angleDeg, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (rotate_host_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::I8, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, angleDeg, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_rotate_u8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *angleDeg, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (rotate_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::U8, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, angleDeg, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_rotate_f16_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *angleDeg, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (rotate_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::FP16, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, angleDeg, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_rotate_f32_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *angleDeg, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (rotate_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::FP32, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, angleDeg, outputFormatToggle, nbatchSize, rppHandle));
}
RppStatus
rppi_rotate_i8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *angleDeg, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (rotate_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::I8, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, angleDeg, outputFormatToggle, nbatchSize, rppHandle));
}

/******************** warp_affine ********************/

RppStatus warp_affine_host_helper(RppiChnFormat chn_format,
                                  Rpp32u num_of_channels,
                                  RPPTensorDataType in_tensor_type,
                                  RPPTensorDataType out_tensor_type,
                                  Rpp8u outputFormatToggle,
                                  RppPtr_t srcPtr,
                                  RppiSize *srcSize,
                                  RppiSize maxSrcSize,
                                  RppPtr_t dstPtr,
                                  RppiSize *dstSize,
                                  RppiSize maxDstSize,
                                  Rpp32f *affineMatrix,
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
    copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));

    if (in_tensor_type == RPPTensorDataType::U8)
    {
        if (out_tensor_type == RPPTensorDataType::U8)
        {
            warp_affine_host_batch<Rpp8u>(static_cast<Rpp8u *>(srcPtr),
                                          srcSize,
                                          rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                          static_cast<Rpp8u *>(dstPtr),
                                          dstSize,
                                          rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
                                          rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                                          affineMatrix,
                                          outputFormatToggle,
                                          rpp::deref(rppHandle).GetBatchSize(),
                                          chn_format,
                                          num_of_channels,
                                          rpp::deref(rppHandle));
        }
    }
    else if (in_tensor_type == RPPTensorDataType::FP16)
    {
        if (out_tensor_type == RPPTensorDataType::FP16)
        {
            warp_affine_host_batch<Rpp16f>(static_cast<Rpp16f *>(srcPtr),
                                           srcSize,
                                           rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                           static_cast<Rpp16f *>(dstPtr),
                                           dstSize,
                                           rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
                                           rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                                           affineMatrix,
                                           outputFormatToggle,
                                           rpp::deref(rppHandle).GetBatchSize(),
                                           chn_format,
                                           num_of_channels,
                                           rpp::deref(rppHandle));
        }
    }
    else if (in_tensor_type == RPPTensorDataType::FP32)
    {
        if (out_tensor_type == RPPTensorDataType::FP32)
        {
            warp_affine_host_batch<Rpp32f>(static_cast<Rpp32f *>(srcPtr),
                                           srcSize,
                                           rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                           static_cast<Rpp32f *>(dstPtr),
                                           dstSize,
                                           rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
                                           rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                                           affineMatrix,
                                           outputFormatToggle,
                                           rpp::deref(rppHandle).GetBatchSize(),
                                           chn_format,
                                           num_of_channels,
                                           rpp::deref(rppHandle));
        }
    }
    else if (in_tensor_type == RPPTensorDataType::I8)
    {
        if (out_tensor_type == RPPTensorDataType::I8)
        {
            warp_affine_host_batch<Rpp8s>(static_cast<Rpp8s *>(srcPtr),
                                          srcSize,
                                          rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                          static_cast<Rpp8s *>(dstPtr),
                                          dstSize,
                                          rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
                                          rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                                          affineMatrix,
                                          outputFormatToggle,
                                          rpp::deref(rppHandle).GetBatchSize(),
                                          chn_format,
                                          num_of_channels,
                                          rpp::deref(rppHandle));
        }
    }

    return RPP_SUCCESS;
}

RppStatus
rppi_warp_affine_u8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *affineMatrix, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return ( warp_affine_host_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::U8, RPPTensorDataType::U8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, affineMatrix, nbatchSize, rppHandle));
}
RppStatus
rppi_warp_affine_f16_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *affineMatrix, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return ( warp_affine_host_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::FP16, RPPTensorDataType::FP16, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, affineMatrix, nbatchSize, rppHandle));
}
RppStatus
rppi_warp_affine_f32_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *affineMatrix, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return ( warp_affine_host_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::FP32, RPPTensorDataType::FP32, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, affineMatrix, nbatchSize, rppHandle));
}
RppStatus
rppi_warp_affine_i8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *affineMatrix, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return ( warp_affine_host_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::I8, RPPTensorDataType::I8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, affineMatrix, nbatchSize, rppHandle));
}
RppStatus
rppi_warp_affine_u8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *affineMatrix, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return ( warp_affine_host_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::U8, RPPTensorDataType::U8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, affineMatrix, nbatchSize, rppHandle));
}
RppStatus
rppi_warp_affine_f16_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *affineMatrix, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return ( warp_affine_host_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::FP16, RPPTensorDataType::FP16, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, affineMatrix, nbatchSize, rppHandle));
}
RppStatus
rppi_warp_affine_f32_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *affineMatrix, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return ( warp_affine_host_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::FP32, RPPTensorDataType::FP32, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, affineMatrix, nbatchSize, rppHandle));
}
RppStatus
rppi_warp_affine_i8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *affineMatrix, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return ( warp_affine_host_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::I8, RPPTensorDataType::I8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, affineMatrix, nbatchSize, rppHandle));
}
RppStatus
rppi_warp_affine_u8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *affineMatrix, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return ( warp_affine_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::U8, RPPTensorDataType::U8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, affineMatrix, nbatchSize, rppHandle));
}
RppStatus
rppi_warp_affine_f16_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *affineMatrix, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return ( warp_affine_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::FP16, RPPTensorDataType::FP16, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, affineMatrix, nbatchSize, rppHandle));
}
RppStatus
rppi_warp_affine_f32_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *affineMatrix, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return ( warp_affine_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::FP32, RPPTensorDataType::FP32, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, affineMatrix, nbatchSize, rppHandle));
}
RppStatus
rppi_warp_affine_i8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *affineMatrix, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return ( warp_affine_host_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::I8, RPPTensorDataType::I8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, affineMatrix, nbatchSize, rppHandle));
}

/******************** fisheye ********************/

RppStatus
rppi_fisheye_u8_pln1_batchPD_host(RppPtr_t srcPtr,
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

    fisheye_host_batch<Rpp8u>(static_cast<Rpp8u *>(srcPtr),
                              srcSize,
                              rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                              static_cast<Rpp8u *>(dstPtr),
                              rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                              rpp::deref(rppHandle).GetBatchSize(),
                              RPPI_CHN_PLANAR,
                              1,
                              rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_fisheye_u8_pln3_batchPD_host(RppPtr_t srcPtr,
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

    fisheye_host_batch<Rpp8u>(static_cast<Rpp8u *>(srcPtr),
                              srcSize,
                              rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                              static_cast<Rpp8u *>(dstPtr),
                              rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                              rpp::deref(rppHandle).GetBatchSize(),
                              RPPI_CHN_PLANAR,
                              3,
                              rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_fisheye_u8_pkd3_batchPD_host(RppPtr_t srcPtr,
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

    fisheye_host_batch<Rpp8u>(static_cast<Rpp8u *>(srcPtr),
                              srcSize,
                              rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                              static_cast<Rpp8u *>(dstPtr),
                              rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                              rpp::deref(rppHandle).GetBatchSize(),
                              RPPI_CHN_PACKED,
                              3,
                              rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

/******************** lens_correction ********************/

RppStatus
rppi_lens_correction_u8_pln1_batchPD_host(RppPtr_t srcPtr,
                                          RppiSize *srcSize,
                                          RppiSize maxSrcSize,
                                          RppPtr_t dstPtr,
                                          Rpp32f *strength,
                                          Rpp32f *zoom,
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

    lens_correction_host_batch<Rpp8u>(static_cast<Rpp8u *>(srcPtr),
                                      srcSize,
                                      rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                      static_cast<Rpp8u *>(dstPtr),
                                      strength,
                                      zoom,
                                      rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                                      rpp::deref(rppHandle).GetBatchSize(),
                                      RPPI_CHN_PLANAR,
                                      1,
                                      rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_lens_correction_u8_pln3_batchPD_host(RppPtr_t srcPtr,
                                          RppiSize *srcSize,
                                          RppiSize maxSrcSize,
                                          RppPtr_t dstPtr,
                                          Rpp32f *strength,
                                          Rpp32f *zoom,
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

    lens_correction_host_batch<Rpp8u>(static_cast<Rpp8u *>(srcPtr),
                                      srcSize,
                                      rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                      static_cast<Rpp8u *>(dstPtr),
                                      strength,
                                      zoom,
                                      rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                                      rpp::deref(rppHandle).GetBatchSize(),
                                      RPPI_CHN_PLANAR,
                                      3,
                                      rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_lens_correction_u8_pkd3_batchPD_host(RppPtr_t srcPtr,
                                          RppiSize *srcSize,
                                          RppiSize maxSrcSize,
                                          RppPtr_t dstPtr,
                                          Rpp32f *strength,
                                          Rpp32f *zoom,
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

    lens_correction_host_batch<Rpp8u>(static_cast<Rpp8u *>(srcPtr),
                                      srcSize,
                                      rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                      static_cast<Rpp8u *>(dstPtr),
                                      strength,
                                      zoom,
                                      rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                                      rpp::deref(rppHandle).GetBatchSize(),
                                      RPPI_CHN_PACKED,
                                      3,
                                      rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

/******************** scale ********************/

RppStatus
rppi_scale_u8_pln1_batchPD_host(RppPtr_t srcPtr,
                                RppiSize *srcSize,
                                RppiSize maxSrcSize,
                                RppPtr_t dstPtr,
                                RppiSize *dstSize,
                                RppiSize maxDstSize,
                                Rpp32f *percentage,
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
    copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));

    scale_host_batch<Rpp8u>(static_cast<Rpp8u *>(srcPtr),
                            srcSize,
                            rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                            static_cast<Rpp8u *>(dstPtr),
                            dstSize,
                            rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
                            percentage,
                            rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                            rpp::deref(rppHandle).GetBatchSize(),
                            RPPI_CHN_PLANAR,
                            1,
                            rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_scale_u8_pln3_batchPD_host(RppPtr_t srcPtr,
                                RppiSize *srcSize,
                                RppiSize maxSrcSize,
                                RppPtr_t dstPtr,
                                RppiSize *dstSize,
                                RppiSize maxDstSize,
                                Rpp32f *percentage,
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
    copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));

    scale_host_batch<Rpp8u>(static_cast<Rpp8u *>(srcPtr),
                            srcSize,
                            rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                            static_cast<Rpp8u *>(dstPtr),
                            dstSize,
                            rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
                            percentage,
                            rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                            rpp::deref(rppHandle).GetBatchSize(),
                            RPPI_CHN_PLANAR,
                            3,
                            rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_scale_u8_pkd3_batchPD_host(RppPtr_t srcPtr,
                                RppiSize *srcSize,
                                RppiSize maxSrcSize,
                                RppPtr_t dstPtr,
                                RppiSize *dstSize,
                                RppiSize maxDstSize,
                                Rpp32f *percentage,
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
    copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));

    scale_host_batch<Rpp8u>(static_cast<Rpp8u *>(srcPtr),
                            srcSize,
                            rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                            static_cast<Rpp8u *>(dstPtr),
                            dstSize,
                            rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
                            percentage,
                            rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                            rpp::deref(rppHandle).GetBatchSize(),
                            RPPI_CHN_PACKED,
                            3,
                            rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

/******************** warp_perspective ********************/

RppStatus
rppi_warp_perspective_u8_pln1_batchPD_host(RppPtr_t srcPtr,
                                           RppiSize *srcSize,
                                           RppiSize maxSrcSize,
                                           RppPtr_t dstPtr,
                                           RppiSize *dstSize,
                                           RppiSize maxDstSize,
                                           Rpp32f *perspectiveMatrix,
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
    copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));

    warp_perspective_host_batch<Rpp8u>(static_cast<Rpp8u *>(srcPtr),
                                       srcSize,
                                       rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                       static_cast<Rpp8u *>(dstPtr),
                                       dstSize,
                                       rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
                                       rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                                       perspectiveMatrix,
                                       rpp::deref(rppHandle).GetBatchSize(),
                                       RPPI_CHN_PLANAR,
                                       1,
                                       rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_warp_perspective_u8_pln3_batchPD_host(RppPtr_t srcPtr,
                                           RppiSize *srcSize,
                                           RppiSize maxSrcSize,
                                           RppPtr_t dstPtr,
                                           RppiSize *dstSize,
                                           RppiSize maxDstSize,
                                           Rpp32f *perspectiveMatrix,
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
    copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));

    warp_perspective_host_batch<Rpp8u>(static_cast<Rpp8u *>(srcPtr),
                                       srcSize,
                                       rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                       static_cast<Rpp8u *>(dstPtr),
                                       dstSize,
                                       rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
                                       rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                                       perspectiveMatrix,
                                       rpp::deref(rppHandle).GetBatchSize(),
                                       RPPI_CHN_PLANAR,
                                       3,
                                       rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus
rppi_warp_perspective_u8_pkd3_batchPD_host(RppPtr_t srcPtr,
                                           RppiSize *srcSize,
                                           RppiSize maxSrcSize,
                                           RppPtr_t dstPtr,
                                           RppiSize *dstSize,
                                           RppiSize maxDstSize,
                                           Rpp32f *perspectiveMatrix,
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
    copy_host_maxDstSize(maxDstSize, rpp::deref(rppHandle));

    warp_perspective_host_batch<Rpp8u>(static_cast<Rpp8u *>(srcPtr),
                                       srcSize,
                                       rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                                       static_cast<Rpp8u *>(dstPtr),
                                       dstSize,
                                       rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxDstSize,
                                       rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                                       perspectiveMatrix,
                                       rpp::deref(rppHandle).GetBatchSize(),
                                       RPPI_CHN_PACKED,
                                       3,
                                       rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

/********************************************************************************************************************/
/*********************************************** RPP_GPU_SUPPORT = ON ***********************************************/
/********************************************************************************************************************/

#ifdef GPU_SUPPORT

/******************** flip ********************/

RppStatus
rppi_flip_u8_pln1_batchPD_gpu(RppPtr_t srcPtr,
                              RppiSize *srcSize,
                              RppiSize maxSrcSize,
                              RppPtr_t dstPtr,
                              Rpp32u *flipAxis,
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
    copy_param_uint(flipAxis, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        flip_cl_batch(static_cast<cl_mem>(srcPtr),
                      static_cast<cl_mem>(dstPtr),
                      rpp::deref(rppHandle),
                      RPPI_CHN_PLANAR,
                      1);
    }
#elif defined(HIP_COMPILE)
    {
        flip_hip_batch(static_cast<Rpp8u *>(srcPtr),
                       static_cast<Rpp8u *>(dstPtr),
                       rpp::deref(rppHandle),
                       RPPI_CHN_PLANAR,
                       1);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_flip_u8_pln3_batchPD_gpu(RppPtr_t srcPtr,
                              RppiSize *srcSize,
                              RppiSize maxSrcSize,
                              RppPtr_t dstPtr,
                              Rpp32u *flipAxis,
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
    copy_param_uint(flipAxis, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        flip_cl_batch(static_cast<cl_mem>(srcPtr),
                      static_cast<cl_mem>(dstPtr),
                      rpp::deref(rppHandle),
                      RPPI_CHN_PLANAR,
                      3);
    }
#elif defined(HIP_COMPILE)
    {
        flip_hip_batch(static_cast<Rpp8u *>(srcPtr),
                       static_cast<Rpp8u *>(dstPtr),
                       rpp::deref(rppHandle),
                       RPPI_CHN_PLANAR,
                       3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_flip_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr,
                              RppiSize *srcSize,
                              RppiSize maxSrcSize,
                              RppPtr_t dstPtr,
                              Rpp32u *flipAxis,
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
    copy_param_uint(flipAxis, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        flip_cl_batch(static_cast<cl_mem>(srcPtr),
                      static_cast<cl_mem>(dstPtr),
                      rpp::deref(rppHandle),
                      RPPI_CHN_PACKED,
                      3);
    }
#elif defined(HIP_COMPILE)
    {
        flip_hip_batch(static_cast<Rpp8u *>(srcPtr),
                       static_cast<Rpp8u *>(dstPtr),
                       rpp::deref(rppHandle),
                       RPPI_CHN_PACKED,
                       3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_flip_u8_pkd4_batchPD_gpu(RppPtr_t  srcPtr,
                              RppiSize *srcSize,
                              RppiSize  maxSrcSize,
                              RppPtr_t  dstPtr,
                              Rpp32u   *flipAxis,
                              Rpp32u    nbatchSize,
                              rppHandle_t rppHandle)
{
    RppiROI roiPoints = {0,0, 0,0};
    Rpp32u paramIndex = 0;

    copy_srcSize(srcSize,       rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    copy_roi(roiPoints,         rpp::deref(rppHandle));

    get_srcBatchIndex(rpp::deref(rppHandle), 4, RPPI_CHN_PACKED);

    copy_param_uint(flipAxis, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    flip_cl_batch(static_cast<cl_mem>(srcPtr),
                  static_cast<cl_mem>(dstPtr),
                  rpp::deref(rppHandle),
                  RPPI_CHN_PACKED, 4);
#elif defined(HIP_COMPILE)
    flip_hip_batch(static_cast<Rpp8u*>(srcPtr),
                   static_cast<Rpp8u*>(dstPtr),
                   rpp::deref(rppHandle),
                   RPPI_CHN_PACKED, 4);
#endif

    return RPP_SUCCESS;
}

RppStatus
rppi_flip_pln1_batchPD_gpu(RppPtr_t srcPtr,
                              RpptDataType dataType,
                              RppiSize *srcSize,
                              RppiSize maxSrcSize,
                              RppPtr_t dstPtr,
                              Rpp32u *flipAxis,
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
    copy_param_uint(flipAxis, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        flip_cl_batch(static_cast<cl_mem>(srcPtr),
                      static_cast<cl_mem>(dstPtr),
                      rpp::deref(rppHandle),
                      RPPI_CHN_PLANAR,
                      1);
    }
#elif defined(HIP_COMPILE)
    {
    switch(dataType)
    {
        case S32: 
            flip_hip_batch_s32(static_cast<Rpp32s *>(srcPtr),
               static_cast<Rpp32s *>(dstPtr),
               rpp::deref(rppHandle),
               RPPI_CHN_PLANAR,
               1);
            break;
        case U16:
            flip_hip_batch_u16(static_cast<Rpp16u *>(srcPtr),
               static_cast<Rpp16u *>(dstPtr),
               rpp::deref(rppHandle),
               RPPI_CHN_PLANAR,
               1);
            break;
        case F32:
            flip_hip_batch_f32(static_cast<Rpp32f *>(srcPtr),
               static_cast<Rpp32f *>(dstPtr),
               rpp::deref(rppHandle),
               RPPI_CHN_PLANAR,
               1);
            break;
    }

    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_flip_pkd3_batchPD_gpu(RppPtr_t srcPtr,
                              RpptDataType dataType,
                              RppiSize *srcSize,
                              RppiSize maxSrcSize,
                              RppPtr_t dstPtr,
                              Rpp32u *flipAxis,
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
    copy_param_uint(flipAxis, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        flip_cl_batch(static_cast<cl_mem>(srcPtr),
                      static_cast<cl_mem>(dstPtr),
                      rpp::deref(rppHandle),
                      RPPI_CHN_PACKED,
                      3);
    }
#elif defined(HIP_COMPILE)
    switch(dataType)
    {
        case S32: 
            flip_hip_batch_s32(static_cast<Rpp32s*>(srcPtr),
                           static_cast<Rpp32s*>(dstPtr),
                           rpp::deref(rppHandle),
                           RPPI_CHN_PACKED, 3);
            break;
        case U16:
            flip_hip_batch_u16(static_cast<Rpp16u*>(srcPtr),
                           static_cast<Rpp16u*>(dstPtr),
                           rpp::deref(rppHandle),
                           RPPI_CHN_PACKED, 3);
            break;
        case F32:
            flip_hip_batch_f32(static_cast<Rpp32f*>(srcPtr),
                           static_cast<Rpp32f*>(dstPtr),
                           rpp::deref(rppHandle),
                           RPPI_CHN_PACKED, 3);
            break;
    }
#endif

    return RPP_SUCCESS;
}

RppStatus
rppi_flip_pkd4_batchPD_gpu(RppPtr_t  srcPtr,
                              RpptDataType dataType,
                              RppiSize *srcSize,
                              RppiSize  maxSrcSize,
                              RppPtr_t  dstPtr,
                              Rpp32u   *flipAxis,
                              Rpp32u    nbatchSize,
                              rppHandle_t rppHandle)
{
    RppiROI roiPoints = {0,0, 0,0};
    Rpp32u paramIndex = 0;

    copy_srcSize(srcSize,       rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    copy_roi(roiPoints,         rpp::deref(rppHandle));

    get_srcBatchIndex(rpp::deref(rppHandle), 4, RPPI_CHN_PACKED);

    copy_param_uint(flipAxis, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    flip_cl_batch(static_cast<cl_mem>(srcPtr),
                  static_cast<cl_mem>(dstPtr),
                  rpp::deref(rppHandle),
                  RPPI_CHN_PACKED, 4);
#elif defined(HIP_COMPILE)
    switch(dataType)
    {
        case S32: 
            flip_hip_batch_s32(static_cast<Rpp32s*>(srcPtr),
                           static_cast<Rpp32s*>(dstPtr),
                           rpp::deref(rppHandle),
                           RPPI_CHN_PACKED, 4);
            break;
        case U16:
            flip_hip_batch_u16(static_cast<Rpp16u*>(srcPtr),
                           static_cast<Rpp16u*>(dstPtr),
                           rpp::deref(rppHandle),
                           RPPI_CHN_PACKED, 4);
            break;
        case F32:
            flip_hip_batch_f32(static_cast<Rpp32f*>(srcPtr),
                           static_cast<Rpp32f*>(dstPtr),
                           rpp::deref(rppHandle),
                           RPPI_CHN_PACKED, 4);
            break;
    }
#endif

    return RPP_SUCCESS;
}                              


/******************** resize ********************/

RppStatus resize_helper(RppiChnFormat chn_format,
                        Rpp32u num_of_channels,
                        RPPTensorDataType in_tensor_type,
                        RPPTensorDataType out_tensor_type,
                        Rpp32u outputFormatToggle,
                        RppPtr_t srcPtr,
                        RppiSize *srcSize,
                        RppiSize maxSrcSize,
                        RppPtr_t dstPtr,
                        RppiSize *dstSize,
                        RppiSize maxDstSize,
                        Rpp32u nbatchSize,
                        rppHandle_t rppHandle)
{
    RPPTensorFunctionMetaData tensor_info(chn_format, in_tensor_type, out_tensor_type, num_of_channels,
                                          (bool)outputFormatToggle);
    RppiROI roiPoints;
    bool is_padded = true;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    copy_dstSize(dstSize, rpp::deref(rppHandle));
    copy_dstMaxSize(maxDstSize, rpp::deref(rppHandle));
    copy_roi(roiPoints, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), num_of_channels, tensor_info._in_format, is_padded);
    get_dstBatchIndex(rpp::deref(rppHandle), num_of_channels, tensor_info._out_format, is_padded);

#ifdef OCL_COMPILE
    {
        resize_cl_batch_tensor(static_cast<cl_mem>(srcPtr),
                               static_cast<cl_mem>(dstPtr),
                               rpp::deref(rppHandle),
                               tensor_info);
    }
#elif defined(HIP_COMPILE)
    {
        if (in_tensor_type == RPPTensorDataType::U8)
        {
            if (out_tensor_type == RPPTensorDataType::U8)
            {
                resize_hip_batch_tensor(static_cast<Rpp8u *>(srcPtr),
                                        static_cast<Rpp8u *>(dstPtr),
                                        rpp::deref(rppHandle),
                                        tensor_info);
            }
            else if (out_tensor_type == RPPTensorDataType::FP16)
            {
                resize_hip_batch_tensor_u8_fp16(static_cast<Rpp8u *>(srcPtr),
                                                static_cast<Rpp16f *>(dstPtr),
                                                rpp::deref(rppHandle),
                                                tensor_info);
            }
            else if (out_tensor_type == RPPTensorDataType::FP32)
            {
                resize_hip_batch_tensor_u8_fp32(static_cast<Rpp8u *>(srcPtr),
                                                static_cast<Rpp32f *>(dstPtr),
                                                rpp::deref(rppHandle),
                                                tensor_info);
            }
            else if (out_tensor_type == RPPTensorDataType::I8)
            {
                resize_hip_batch_tensor_u8_int8(static_cast<Rpp8u *>(srcPtr),
                                                static_cast<Rpp8s *>(dstPtr),
                                                rpp::deref(rppHandle),
                                                tensor_info);
            }
        }
        else if (in_tensor_type == RPPTensorDataType::FP16)
        {
            resize_hip_batch_tensor_fp16(static_cast<Rpp16f *>(srcPtr),
                                         static_cast<Rpp16f *>(dstPtr),
                                         rpp::deref(rppHandle),
                                         tensor_info);
        }
        else if (in_tensor_type == RPPTensorDataType::FP32)
        {
            resize_hip_batch_tensor_fp32(static_cast<Rpp32f *>(srcPtr),
                                         static_cast<Rpp32f *>(dstPtr),
                                         rpp::deref(rppHandle),
                                         tensor_info);
        }
        else if (in_tensor_type == RPPTensorDataType::I8)
        {
            resize_hip_batch_tensor_int8(static_cast<Rpp8s *>(srcPtr),
                                         static_cast<Rpp8s *>(dstPtr),
                                         rpp::deref(rppHandle),
                                         tensor_info);
        }
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_resize_u8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::U8, RPPTensorDataType::U8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxSrcSize, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_f16_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::FP16, RPPTensorDataType::FP16, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxSrcSize, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_f32_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::FP32, RPPTensorDataType::FP32, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxSrcSize, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_i8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::I8, RPPTensorDataType::I8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxSrcSize, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_u8_f16_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::U8, RPPTensorDataType::FP16, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxSrcSize, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_u8_f32_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::U8, RPPTensorDataType::FP32, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxSrcSize, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_u8_i8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::U8, RPPTensorDataType::I8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxSrcSize, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_u8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::U8, RPPTensorDataType::U8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxSrcSize, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_f16_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::FP16, RPPTensorDataType::FP16, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxSrcSize, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_f32_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::FP32, RPPTensorDataType::FP32, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxSrcSize, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_i8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::I8, RPPTensorDataType::I8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxSrcSize, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_u8_f16_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::U8, RPPTensorDataType::FP16, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxSrcSize, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_u8_f32_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::U8, RPPTensorDataType::FP32, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxSrcSize, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_u8_i8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::U8, RPPTensorDataType::I8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxSrcSize, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::U8, RPPTensorDataType::U8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_f16_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::FP16, RPPTensorDataType::FP16, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxSrcSize, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_f32_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::FP32, RPPTensorDataType::FP32, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxSrcSize, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_i8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::I8, RPPTensorDataType::I8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxSrcSize, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_u8_f16_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::U8, RPPTensorDataType::FP16, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxSrcSize, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_u8_f32_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::U8, RPPTensorDataType::FP32, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxSrcSize, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_u8_i8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::U8, RPPTensorDataType::I8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxSrcSize, nbatchSize, rppHandle));
}

/******************** resize_crop ********************/

RppStatus resize_crop_helper(RppiChnFormat chn_format,
                             Rpp32u num_of_channels,
                             RPPTensorDataType in_tensor_type,
                             RPPTensorDataType out_tensor_type,
                             Rpp8u outputFormatToggle,
                             RppPtr_t srcPtr,
                             RppiSize *srcSize,
                             RppiSize maxSrcSize,
                             RppPtr_t dstPtr,
                             RppiSize *dstSize,
                             RppiSize maxDstSize,
                             Rpp32u *xRoiBegin,
                             Rpp32u *xRoiEnd,
                             Rpp32u *yRoiBegin,
                             Rpp32u *yRoiEnd,
                             Rpp32u nbatchSize,
                             rppHandle_t rppHandle)
{
    RPPTensorFunctionMetaData tensor_info(chn_format, in_tensor_type, out_tensor_type, num_of_channels,
                                          (bool)outputFormatToggle);
    Rpp32u paramIndex = 0;
    bool is_padded = true;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    copy_dstSize(dstSize, rpp::deref(rppHandle));
    copy_dstMaxSize(maxDstSize, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), num_of_channels, tensor_info._in_format, is_padded);
    get_dstBatchIndex(rpp::deref(rppHandle), num_of_channels, tensor_info._out_format, is_padded);
    copy_param_uint(xRoiBegin, rpp::deref(rppHandle), paramIndex++);
    copy_param_uint(xRoiEnd, rpp::deref(rppHandle), paramIndex++);
    copy_param_uint(yRoiBegin, rpp::deref(rppHandle), paramIndex++);
    copy_param_uint(yRoiEnd, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        resize_crop_cl_batch_tensor(static_cast<cl_mem>(srcPtr),
                                    static_cast<cl_mem>(dstPtr),
                                    rpp::deref(rppHandle),
                                    tensor_info);
    }
#elif defined(HIP_COMPILE)
    {
        if (in_tensor_type == RPPTensorDataType::U8)
        {
            if (out_tensor_type == RPPTensorDataType::U8)
            {
                resize_crop_hip_batch_tensor(static_cast<Rpp8u *>(srcPtr),
                                             static_cast<Rpp8u *>(dstPtr),
                                             rpp::deref(rppHandle),
                                             tensor_info);
            }
            else if (out_tensor_type == RPPTensorDataType::FP16)
            {
                resize_crop_hip_batch_tensor_u8_fp16(static_cast<Rpp8u *>(srcPtr),
                                                     static_cast<Rpp16f *>(dstPtr),
                                                     rpp::deref(rppHandle),
                                                     tensor_info);
            }
            else if (out_tensor_type == RPPTensorDataType::FP32)
            {
                resize_crop_hip_batch_tensor_u8_fp32(static_cast<Rpp8u *>(srcPtr),
                                                     static_cast<Rpp32f *>(dstPtr),
                                                     rpp::deref(rppHandle),
                                                     tensor_info);
            }
            else if (out_tensor_type == RPPTensorDataType::I8)
            {
                resize_crop_hip_batch_tensor_u8_int8(static_cast<Rpp8u *>(srcPtr),
                                                     static_cast<Rpp8s *>(dstPtr),
                                                     rpp::deref(rppHandle),
                                                     tensor_info);
            }
        }
        else if (in_tensor_type == RPPTensorDataType::FP16)
        {
            resize_crop_hip_batch_tensor_fp16(static_cast<Rpp16f *>(srcPtr),
                                              static_cast<Rpp16f *>(dstPtr),
                                              rpp::deref(rppHandle),
                                              tensor_info);
        }
        else if (in_tensor_type == RPPTensorDataType::FP32)
        {
            resize_crop_hip_batch_tensor_fp32(static_cast<Rpp32f *>(srcPtr),
                                              static_cast<Rpp32f *>(dstPtr),
                                              rpp::deref(rppHandle),
                                              tensor_info);
        }
        else if (in_tensor_type == RPPTensorDataType::I8)
        {
            resize_crop_hip_batch_tensor_int8(static_cast<Rpp8s *>(srcPtr),
                                              static_cast<Rpp8s *>(dstPtr),
                                              rpp::deref(rppHandle),
                                              tensor_info);
        }
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_resize_crop_u8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_crop_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::U8, RPPTensorDataType::U8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, xRoiBegin, xRoiEnd, yRoiBegin, yRoiEnd, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_crop_f16_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_crop_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::FP16, RPPTensorDataType::FP16, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, xRoiBegin, xRoiEnd, yRoiBegin, yRoiEnd, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_crop_f32_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_crop_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::FP32, RPPTensorDataType::FP32, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, xRoiBegin, xRoiEnd, yRoiBegin, yRoiEnd, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_crop_i8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_crop_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::I8, RPPTensorDataType::I8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, xRoiBegin, xRoiEnd, yRoiBegin, yRoiEnd, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_crop_u8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_crop_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::U8, RPPTensorDataType::U8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, xRoiBegin, xRoiEnd, yRoiBegin, yRoiEnd, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_crop_f16_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_crop_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::FP16, RPPTensorDataType::FP16, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, xRoiBegin, xRoiEnd, yRoiBegin, yRoiEnd, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_crop_f32_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_crop_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::FP32, RPPTensorDataType::FP32, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, xRoiBegin, xRoiEnd, yRoiBegin, yRoiEnd, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_crop_i8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_crop_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::I8, RPPTensorDataType::I8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, xRoiBegin, xRoiEnd, yRoiBegin, yRoiEnd, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_crop_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_crop_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::U8, RPPTensorDataType::U8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, xRoiBegin, xRoiEnd, yRoiBegin, yRoiEnd, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_crop_f16_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_crop_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::FP16, RPPTensorDataType::FP16, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, xRoiBegin, xRoiEnd, yRoiBegin, yRoiEnd, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_crop_f32_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_crop_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::FP32, RPPTensorDataType::FP32, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, xRoiBegin, xRoiEnd, yRoiBegin, yRoiEnd, nbatchSize, rppHandle));
}
RppStatus
rppi_resize_crop_i8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (resize_crop_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::I8, RPPTensorDataType::I8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, xRoiBegin, xRoiEnd, yRoiBegin, yRoiEnd, nbatchSize, rppHandle));
}

/******************** rotate ********************/

RppStatus rotate_helper(RppiChnFormat chn_format,
                        Rpp32u num_of_channels,
                        RPPTensorDataType in_tensor_type,
                        RPPTensorDataType out_tensor_type,
                        Rpp32u outputFormatToggle,
                        RppPtr_t srcPtr,
                        RppiSize *srcSize,
                        RppiSize maxSrcSize,
                        RppPtr_t dstPtr,
                        RppiSize *dstSize,
                        RppiSize maxDstSize,
                        Rpp32f *angleDeg,
                        Rpp32u nbatchSize,
                        rppHandle_t rppHandle)
{
    RPPTensorFunctionMetaData tensor_info(chn_format, in_tensor_type, out_tensor_type, num_of_channels,
                                          (bool)outputFormatToggle);
    RppiROI roiPoints;
    bool is_padded = true;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    Rpp32u paramIndex = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    copy_dstSize(dstSize, rpp::deref(rppHandle));
    copy_dstMaxSize(maxDstSize, rpp::deref(rppHandle));
    copy_roi(roiPoints, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), num_of_channels, tensor_info._in_format, is_padded);
    get_dstBatchIndex(rpp::deref(rppHandle), num_of_channels, tensor_info._out_format, is_padded);
    copy_param_float(angleDeg, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        rotate_cl_batch_tensor(static_cast<cl_mem>(srcPtr),
                               static_cast<cl_mem>(dstPtr),
                               rpp::deref(rppHandle),
                               tensor_info);
    }
#elif defined(HIP_COMPILE)
    {
        if (in_tensor_type == RPPTensorDataType::U8)
        {
            rotate_hip_batch_tensor(static_cast<Rpp8u *>(srcPtr),
                                    static_cast<Rpp8u *>(dstPtr),
                                    rpp::deref(rppHandle),
                                    tensor_info);
        }
        else if (in_tensor_type == RPPTensorDataType::FP16)
        {
            rotate_hip_batch_tensor_fp16(static_cast<Rpp16f *>(srcPtr),
                                         static_cast<Rpp16f *>(dstPtr),
                                         rpp::deref(rppHandle),
                                         tensor_info);
        }
        else if (in_tensor_type == RPPTensorDataType::FP32)
        {
            rotate_hip_batch_tensor_fp32(static_cast<Rpp32f *>(srcPtr),
                                         static_cast<Rpp32f *>(dstPtr),
                                         rpp::deref(rppHandle),
                                         tensor_info);
        }
        else if (in_tensor_type == RPPTensorDataType::I8)
        {
            rotate_hip_batch_tensor_int8(static_cast<Rpp8s *>(srcPtr),
                                         static_cast<Rpp8s *>(dstPtr),
                                         rpp::deref(rppHandle),
                                         tensor_info);
        }
    }
#endif //BACKEND
    return RPP_SUCCESS;
}

RppStatus
rppi_rotate_u8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *angleDeg, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (rotate_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::U8, RPPTensorDataType::U8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, angleDeg, nbatchSize, rppHandle));
}
RppStatus
rppi_rotate_f16_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *angleDeg, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (rotate_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::FP16, RPPTensorDataType::FP16, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, angleDeg, nbatchSize, rppHandle));
}
RppStatus
rppi_rotate_f32_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *angleDeg, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (rotate_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::FP32, RPPTensorDataType::FP32, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, angleDeg, nbatchSize, rppHandle));
}
RppStatus
rppi_rotate_i8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *angleDeg, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (rotate_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::I8, RPPTensorDataType::I8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, angleDeg, nbatchSize, rppHandle));
}
RppStatus
rppi_rotate_u8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *angleDeg, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (rotate_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::U8, RPPTensorDataType::U8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, angleDeg, nbatchSize, rppHandle));
}
RppStatus
rppi_rotate_f16_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *angleDeg, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (rotate_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::FP16, RPPTensorDataType::FP16, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, angleDeg, nbatchSize, rppHandle));
}
RppStatus
rppi_rotate_f32_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *angleDeg, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (rotate_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::FP32, RPPTensorDataType::FP32, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, angleDeg, nbatchSize, rppHandle));
}
RppStatus
rppi_rotate_i8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *angleDeg, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (rotate_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::I8, RPPTensorDataType::I8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, angleDeg, nbatchSize, rppHandle));
}
RppStatus
rppi_rotate_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *angleDeg, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (rotate_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::U8, RPPTensorDataType::U8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, angleDeg, nbatchSize, rppHandle));
}
RppStatus
rppi_rotate_f16_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *angleDeg, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (rotate_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::FP16, RPPTensorDataType::FP16, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, angleDeg, nbatchSize, rppHandle));
}
RppStatus
rppi_rotate_f32_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *angleDeg, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (rotate_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::FP32, RPPTensorDataType::FP32, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, angleDeg, nbatchSize, rppHandle));
}
RppStatus
rppi_rotate_i8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *angleDeg, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (rotate_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::I8, RPPTensorDataType::I8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, angleDeg, nbatchSize, rppHandle));
}

/******************** warp_affine ********************/

RppStatus warp_affine_helper(RppiChnFormat chn_format,
                             Rpp32u num_of_channels,
                             RPPTensorDataType in_tensor_type,
                             RPPTensorDataType out_tensor_type,
                             Rpp32u outputFormatToggle,
                             RppPtr_t srcPtr,
                             RppiSize *srcSize,
                             RppiSize maxSrcSize,
                             RppPtr_t dstPtr,
                             RppiSize *dstSize,
                             RppiSize maxDstSize,
                             Rpp32f *affineMatrix,
                             Rpp32u nbatchSize,
                             rppHandle_t rppHandle)
{
    RPPTensorFunctionMetaData tensor_info(chn_format, in_tensor_type, out_tensor_type, num_of_channels,
                                          (bool)outputFormatToggle);
    RppiROI roiPoints;
    bool is_padded = true;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    copy_dstSize(dstSize, rpp::deref(rppHandle));
    copy_dstMaxSize(maxDstSize, rpp::deref(rppHandle));
    copy_roi(roiPoints, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), num_of_channels, tensor_info._in_format, is_padded);
    get_dstBatchIndex(rpp::deref(rppHandle), num_of_channels, tensor_info._out_format, is_padded);

#ifdef OCL_COMPILE
    {
        warp_affine_cl_batch_tensor(static_cast<cl_mem>(srcPtr),
                                    static_cast<cl_mem>(dstPtr),
                                    rpp::deref(rppHandle), affineMatrix,
                                    tensor_info);
    }
#elif defined(HIP_COMPILE)
    {
        if (in_tensor_type == RPPTensorDataType::U8)
        {
            warp_affine_hip_batch_tensor(static_cast<Rpp8u *>(srcPtr),
                                         static_cast<Rpp8u *>(dstPtr),
                                         rpp::deref(rppHandle),
                                         affineMatrix,
                                         tensor_info);
        }
        else if (in_tensor_type == RPPTensorDataType::FP16)
        {
            warp_affine_hip_batch_tensor_fp16(static_cast<Rpp16f *>(srcPtr),
                                              static_cast<Rpp16f *>(dstPtr),
                                              rpp::deref(rppHandle),
                                              affineMatrix,
                                              tensor_info);
        }
        else if (in_tensor_type == RPPTensorDataType::FP32)
        {
            warp_affine_hip_batch_tensor_fp32(static_cast<Rpp32f *>(srcPtr),
                                              static_cast<Rpp32f *>(dstPtr),
                                              rpp::deref(rppHandle),
                                              affineMatrix,
                                              tensor_info);
        }
        else if (in_tensor_type == RPPTensorDataType::I8)
        {
            warp_affine_hip_batch_tensor_int8(static_cast<Rpp8s *>(srcPtr),
                                              static_cast<Rpp8s *>(dstPtr),
                                              rpp::deref(rppHandle),
                                              affineMatrix,
                                              tensor_info);
        }
    }
#endif //BACKEND

    return RPP_SUCCESS;

}

RppStatus
rppi_warp_affine_u8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *affineMatrix, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (warp_affine_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::U8, RPPTensorDataType::U8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, affineMatrix, nbatchSize, rppHandle));
}
RppStatus
rppi_warp_affine_f16_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *affineMatrix, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (warp_affine_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::FP16, RPPTensorDataType::FP16, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, affineMatrix, nbatchSize, rppHandle));
}
RppStatus
rppi_warp_affine_f32_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *affineMatrix, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (warp_affine_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::FP32, RPPTensorDataType::FP32, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, affineMatrix, nbatchSize, rppHandle));
}
RppStatus
rppi_warp_affine_i8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *affineMatrix, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (warp_affine_helper(RPPI_CHN_PLANAR, 1, RPPTensorDataType::I8, RPPTensorDataType::I8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, affineMatrix, nbatchSize, rppHandle));
}
RppStatus
rppi_warp_affine_u8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *affineMatrix, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (warp_affine_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::U8, RPPTensorDataType::U8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, affineMatrix, nbatchSize, rppHandle));
}
RppStatus
rppi_warp_affine_f16_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *affineMatrix, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (warp_affine_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::FP16, RPPTensorDataType::FP16, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, affineMatrix, nbatchSize, rppHandle));
}
RppStatus
rppi_warp_affine_f32_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *affineMatrix, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (warp_affine_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::FP32, RPPTensorDataType::FP32, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, affineMatrix, nbatchSize, rppHandle));
}
RppStatus
rppi_warp_affine_i8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *affineMatrix, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (warp_affine_helper(RPPI_CHN_PLANAR, 3, RPPTensorDataType::I8, RPPTensorDataType::I8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, affineMatrix, nbatchSize, rppHandle));
}
RppStatus
rppi_warp_affine_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *affineMatrix, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (warp_affine_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::U8, RPPTensorDataType::U8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, affineMatrix, nbatchSize, rppHandle));
}
RppStatus
rppi_warp_affine_f16_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *affineMatrix, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (warp_affine_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::FP16, RPPTensorDataType::FP16, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, affineMatrix, nbatchSize, rppHandle));
}
RppStatus
rppi_warp_affine_f32_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *affineMatrix, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (warp_affine_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::FP32, RPPTensorDataType::FP32, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, affineMatrix, nbatchSize, rppHandle));
}
RppStatus
rppi_warp_affine_i8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *affineMatrix, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle)
{
    return (warp_affine_helper(RPPI_CHN_PACKED, 3, RPPTensorDataType::I8, RPPTensorDataType::I8, outputFormatToggle, srcPtr, srcSize, maxSrcSize, dstPtr, dstSize, maxDstSize, affineMatrix, nbatchSize, rppHandle));
}

/******************** fisheye ********************/

RppStatus
rppi_fisheye_u8_pln1_batchPD_gpu(RppPtr_t srcPtr,
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
        fisheye_cl_batch(static_cast<cl_mem>(srcPtr),
                         static_cast<cl_mem>(dstPtr),
                         rpp::deref(rppHandle),
                         RPPI_CHN_PLANAR,
                         1);
    }
#elif defined(HIP_COMPILE)
    {
        fisheye_hip_batch(static_cast<Rpp8u *>(srcPtr),
                          static_cast<Rpp8u *>(dstPtr),
                          rpp::deref(rppHandle),
                          RPPI_CHN_PLANAR,
                          1);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_fisheye_u8_pln3_batchPD_gpu(RppPtr_t srcPtr,
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
        fisheye_cl_batch(static_cast<cl_mem>(srcPtr),
                         static_cast<cl_mem>(dstPtr),
                         rpp::deref(rppHandle),
                         RPPI_CHN_PLANAR,
                         3);
    }
#elif defined(HIP_COMPILE)
    {
        fisheye_hip_batch(static_cast<Rpp8u *>(srcPtr),
                          static_cast<Rpp8u *>(dstPtr),
                          rpp::deref(rppHandle),
                          RPPI_CHN_PLANAR,
                          3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_fisheye_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr,
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
        fisheye_cl_batch(static_cast<cl_mem>(srcPtr),
                         static_cast<cl_mem>(dstPtr),
                         rpp::deref(rppHandle),
                         RPPI_CHN_PACKED,
                         3);
    }
#elif defined(HIP_COMPILE)
    {
        fisheye_hip_batch(static_cast<Rpp8u *>(srcPtr),
                          static_cast<Rpp8u *>(dstPtr),
                          rpp::deref(rppHandle),
                          RPPI_CHN_PACKED,
                          3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

/******************** lens_correction ********************/

RppStatus
rppi_lens_correction_u8_pln1_batchPD_gpu(RppPtr_t srcPtr,
                                         RppiSize *srcSize,
                                         RppiSize maxSrcSize,
                                         RppPtr_t dstPtr,
                                         Rpp32f *strength,
                                         Rpp32f *zoom,
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
    copy_param_float(strength, rpp::deref(rppHandle), paramIndex++);
    copy_param_float(zoom, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        lens_correction_cl_batch(static_cast<cl_mem>(srcPtr),
                                 static_cast<cl_mem>(dstPtr),
                                 rpp::deref(rppHandle),
                                 RPPI_CHN_PLANAR,
                                 1);
    }
#elif defined(HIP_COMPILE)
    {
        lens_correction_hip_batch(static_cast<Rpp8u *>(srcPtr),
                                  static_cast<Rpp8u *>(dstPtr),
                                  rpp::deref(rppHandle),
                                  RPPI_CHN_PLANAR,
                                  1);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_lens_correction_u8_pln3_batchPD_gpu(RppPtr_t srcPtr,
                                         RppiSize *srcSize,
                                         RppiSize maxSrcSize,
                                         RppPtr_t dstPtr,
                                         Rpp32f *strength,
                                         Rpp32f *zoom,
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
    copy_param_float(strength, rpp::deref(rppHandle), paramIndex++);
    copy_param_float(zoom, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        lens_correction_cl_batch(static_cast<cl_mem>(srcPtr),
                                 static_cast<cl_mem>(dstPtr),
                                 rpp::deref(rppHandle),
                                 RPPI_CHN_PLANAR,
                                 3);
    }
#elif defined(HIP_COMPILE)
    {
        lens_correction_hip_batch(static_cast<Rpp8u *>(srcPtr),
                                  static_cast<Rpp8u *>(dstPtr),
                                  rpp::deref(rppHandle),
                                  RPPI_CHN_PLANAR,
                                  3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_lens_correction_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr,
                                         RppiSize *srcSize,
                                         RppiSize maxSrcSize,
                                         RppPtr_t dstPtr,
                                         Rpp32f *strength,
                                         Rpp32f *zoom,
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
    copy_param_float(strength, rpp::deref(rppHandle), paramIndex++);
    copy_param_float(zoom, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        lens_correction_cl_batch(static_cast<cl_mem>(srcPtr),
                                 static_cast<cl_mem>(dstPtr),
                                 rpp::deref(rppHandle),
                                 RPPI_CHN_PACKED,
                                 3);
    }
#elif defined(HIP_COMPILE)
    {
        lens_correction_hip_batch(static_cast<Rpp8u *>(srcPtr),
                                  static_cast<Rpp8u *>(dstPtr),
                                  rpp::deref(rppHandle),
                                  RPPI_CHN_PACKED,
                                  3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

/******************** scale ********************/

RppStatus
rppi_scale_u8_pln1_batchPD_gpu(RppPtr_t srcPtr,
                               RppiSize *srcSize,
                               RppiSize maxSrcSize,
                               RppPtr_t dstPtr,
                               RppiSize *dstSize,
                               RppiSize maxDstSize,
                               Rpp32f *percentage,
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
    copy_dstSize(dstSize, rpp::deref(rppHandle));
    copy_dstMaxSize(maxDstSize, rpp::deref(rppHandle));
    copy_roi(roiPoints, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
    get_dstBatchIndex(rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
    copy_param_float(percentage, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        scale_cl_batch(static_cast<cl_mem>(srcPtr),
                       static_cast<cl_mem>(dstPtr),
                       rpp::deref(rppHandle),
                       RPPI_CHN_PLANAR,
                       1);
    }
#elif defined(HIP_COMPILE)
    {
        scale_hip_batch(static_cast<Rpp8u *>(srcPtr),
                        static_cast<Rpp8u *>(dstPtr),
                        rpp::deref(rppHandle),
                        RPPI_CHN_PLANAR,
                        1);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_scale_u8_pln3_batchPD_gpu(RppPtr_t srcPtr,
                               RppiSize *srcSize,
                               RppiSize maxSrcSize,
                               RppPtr_t dstPtr,
                               RppiSize *dstSize,
                               RppiSize maxDstSize,
                               Rpp32f *percentage,
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
    copy_dstSize(dstSize, rpp::deref(rppHandle));
    copy_dstMaxSize(maxDstSize, rpp::deref(rppHandle));
    copy_roi(roiPoints, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
    get_dstBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
    copy_param_float(percentage, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        scale_cl_batch(static_cast<cl_mem>(srcPtr),
                       static_cast<cl_mem>(dstPtr),
                       rpp::deref(rppHandle),
                       RPPI_CHN_PLANAR,
                       3);
    }
#elif defined(HIP_COMPILE)
    {
        scale_hip_batch(static_cast<Rpp8u *>(srcPtr),
                        static_cast<Rpp8u *>(dstPtr),
                        rpp::deref(rppHandle),
                        RPPI_CHN_PLANAR,
                        3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_scale_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr,
                               RppiSize *srcSize,
                               RppiSize maxSrcSize,
                               RppPtr_t dstPtr,
                               RppiSize *dstSize,
                               RppiSize maxDstSize,
                               Rpp32f *percentage,
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
    copy_dstSize(dstSize, rpp::deref(rppHandle));
    copy_dstMaxSize(maxDstSize, rpp::deref(rppHandle));
    copy_roi(roiPoints, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
    get_dstBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
    copy_param_float(percentage, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
    {
        scale_cl_batch(static_cast<cl_mem>(srcPtr),
                       static_cast<cl_mem>(dstPtr),
                       rpp::deref(rppHandle),
                       RPPI_CHN_PACKED,
                       3);
    }
#elif defined(HIP_COMPILE)
    {
        scale_hip_batch(static_cast<Rpp8u *>(srcPtr),
                        static_cast<Rpp8u *>(dstPtr),
                        rpp::deref(rppHandle),
                        RPPI_CHN_PACKED,
                        3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

/******************** warp_perspective ********************/

RppStatus
rppi_warp_perspective_u8_pln1_batchPD_gpu(RppPtr_t srcPtr,
                                          RppiSize *srcSize,
                                          RppiSize maxSrcSize,
                                          RppPtr_t dstPtr,
                                          RppiSize *dstSize,
                                          RppiSize maxDstSize,
                                          Rpp32f *perspectiveMatrix,
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
    copy_dstSize(dstSize, rpp::deref(rppHandle));
    copy_dstMaxSize(maxDstSize, rpp::deref(rppHandle));
    copy_roi(roiPoints, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
    get_dstBatchIndex(rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
    {
        warp_perspective_cl_batch(static_cast<cl_mem>(srcPtr),
                                  static_cast<cl_mem>(dstPtr),
                                  rpp::deref(rppHandle),
                                  perspectiveMatrix,
                                  RPPI_CHN_PLANAR,
                                  1);
    }
#elif defined(HIP_COMPILE)
    {
        warp_perspective_hip_batch(static_cast<Rpp8u *>(srcPtr),
                                   static_cast<Rpp8u *>(dstPtr),
                                   rpp::deref(rppHandle),
                                   perspectiveMatrix,
                                   RPPI_CHN_PLANAR,
                                   1);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_warp_perspective_u8_pln3_batchPD_gpu(RppPtr_t srcPtr,
                                          RppiSize *srcSize,
                                          RppiSize maxSrcSize,
                                          RppPtr_t dstPtr,
                                          RppiSize *dstSize,
                                          RppiSize maxDstSize,
                                          Rpp32f *perspectiveMatrix,
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
    copy_dstSize(dstSize, rpp::deref(rppHandle));
    copy_dstMaxSize(maxDstSize, rpp::deref(rppHandle));
    copy_roi(roiPoints, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
    get_dstBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
    {
        warp_perspective_cl_batch(static_cast<cl_mem>(srcPtr),
                                  static_cast<cl_mem>(dstPtr),
                                  rpp::deref(rppHandle),
                                  perspectiveMatrix,
                                  RPPI_CHN_PLANAR,
                                  3);
    }
#elif defined(HIP_COMPILE)
    {
        warp_perspective_hip_batch(static_cast<Rpp8u *>(srcPtr),
                                   static_cast<Rpp8u *>(dstPtr),
                                   rpp::deref(rppHandle),
                                   perspectiveMatrix,
                                   RPPI_CHN_PLANAR,
                                   3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppi_warp_perspective_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr,
                                          RppiSize *srcSize,
                                          RppiSize maxSrcSize,
                                          RppPtr_t dstPtr,
                                          RppiSize *dstSize,
                                          RppiSize maxDstSize,
                                          Rpp32f *perspectiveMatrix,
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
    copy_dstSize(dstSize, rpp::deref(rppHandle));
    copy_dstMaxSize(maxDstSize, rpp::deref(rppHandle));
    copy_roi(roiPoints, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
    get_dstBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
    {
        warp_perspective_cl_batch(static_cast<cl_mem>(srcPtr),
                                  static_cast<cl_mem>(dstPtr),
                                  rpp::deref(rppHandle),
                                  perspectiveMatrix,
                                  RPPI_CHN_PACKED,
                                  3);
    }
#elif defined(HIP_COMPILE)
    {
        warp_perspective_hip_batch(static_cast<Rpp8u *>(srcPtr),
                                   static_cast<Rpp8u *>(dstPtr),
                                   rpp::deref(rppHandle),
                                   perspectiveMatrix,
                                   RPPI_CHN_PACKED,
                                   3);
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

NppStatus nppiWarpPerspective_8u_C3R(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
				Npp8u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3], int eInterpolation)
{
    int noOfImages = 1;
    int ip_channel = 3;//pkd_3
    RppiSize *srcSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize));
	RppiSize *dstSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize));
    RppiSize maxSize,maxDstSize;
    srcSize->width  = oSrcSize.width;
    srcSize->height = oSrcSize.height;
	dstSize->width  = oSrcSize.width;
    dstSize->height = oSrcSize.height;
    maxSize.width  = oSrcROI.width;
    maxSize.height = oSrcROI.height;
	maxDstSize.width = oSrcROI.width;
	maxDstSize.height = oSrcROI.height;
	RppStatus status;
		
	//Rpp32f perspective[9] = {aCoeffs[0][0],aCoeffs[0][1],aCoeffs[0][2],aCoeffs[1][0],aCoeffs[1][1],
	//aCoeffs[1][2],aCoeffs[2][0],aCoeffs[2][1],aCoeffs[2][2]};
	Rpp32f perspective[9] = {0};

    perspective[0] = (Rpp32f)aCoeffs[0][0];
    perspective[1] = (Rpp32f)aCoeffs[0][1];
    perspective[2] = (Rpp32f)aCoeffs[0][2];
    perspective[3] = (Rpp32f)aCoeffs[1][0];
    perspective[4] = (Rpp32f)aCoeffs[1][1];
    perspective[5] = (Rpp32f)aCoeffs[1][2];
    perspective[6] = (Rpp32f)aCoeffs[2][0];
    perspective[7] = (Rpp32f)aCoeffs[2][1];
    perspective[8] = (Rpp32f)aCoeffs[2][2];

    rppHandle_t handle;
    hipStream_t stream;
    hipStreamCreate(&stream);
    rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

	status = rppi_warp_perspective_u8_pkd3_batchPD_gpu((RppPtr_t)pSrc, srcSize, maxSize, (RppPtr_t)pDst, dstSize, maxDstSize, perspective, noOfImages, handle);
    hipDeviceSynchronize();

    rppDestroyGPU(handle);
    free(srcSize);
	free(dstSize);

    return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiWarpPerspective_8u_C1R_Ctx(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp8u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3], int eInterpolation, NppStreamContext nppStreamCtx)
{
        int noOfImages = 1;
        int ip_channel = 1;//pln1
        RppiSize *srcSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize));
        RppiSize *dstSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize));
        RppiSize maxSize,maxDstSize;
        srcSize->width  = oSrcSize.width;
        srcSize->height = oSrcSize.height;
        dstSize->width  = oSrcSize.width;
        dstSize->height = oSrcSize.height;
        maxSize.width  = oSrcROI.width;
        maxSize.height = oSrcROI.height;
        maxDstSize.width = oSrcROI.width;
        maxDstSize.height = oSrcROI.height;
        RppStatus status;

        //Rpp32f perspective[9] = {aCoeffs[0][0],aCoeffs[0][1],aCoeffs[0][2],aCoeffs[1][0],aCoeffs[1][1],
        //aCoeffs[1][2],aCoeffs[2][0],aCoeffs[2][1],aCoeffs[2][2]};
        Rpp32f perspective[9] = {0};

        perspective[0] = (Rpp32f)aCoeffs[0][0];
        perspective[1] = (Rpp32f)aCoeffs[0][1];
        perspective[2] = (Rpp32f)aCoeffs[0][2];
        perspective[3] = (Rpp32f)aCoeffs[1][0];
        perspective[4] = (Rpp32f)aCoeffs[1][1];
        perspective[5] = (Rpp32f)aCoeffs[1][2];
        perspective[6] = (Rpp32f)aCoeffs[2][0];
        perspective[7] = (Rpp32f)aCoeffs[2][1];
        perspective[8] = (Rpp32f)aCoeffs[2][2];

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        status = rppi_warp_perspective_u8_pln1_batchPD_gpu((RppPtr_t)pSrc, srcSize, maxSize, (RppPtr_t)pDst, dstSize, maxDstSize, perspective, noOfImages, handle);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        free(srcSize);
        free(dstSize);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiWarpPerspective_8u_C1R(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp8u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3], int eInterpolation)
{
        int noOfImages = 1;
        int ip_channel = 1;//pln1
        RppiSize *srcSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize));
        RppiSize *dstSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize));
        RppiSize maxSize,maxDstSize;
        srcSize->width  = oSrcSize.width;
        srcSize->height = oSrcSize.height;
        dstSize->width  = oSrcSize.width;
        dstSize->height = oSrcSize.height;
        maxSize.width  = oSrcROI.width;
        maxSize.height = oSrcROI.height;
        maxDstSize.width = oSrcROI.width;
        maxDstSize.height = oSrcROI.height;
        RppStatus status;

        //Rpp32f perspective[9] = {aCoeffs[0][0],aCoeffs[0][1],aCoeffs[0][2],aCoeffs[1][0],aCoeffs[1][1],
        //aCoeffs[1][2],aCoeffs[2][0],aCoeffs[2][1],aCoeffs[2][2]};
        Rpp32f perspective[9] = {0};

        perspective[0] = (Rpp32f)aCoeffs[0][0];
        perspective[1] = (Rpp32f)aCoeffs[0][1];
        perspective[2] = (Rpp32f)aCoeffs[0][2];
        perspective[3] = (Rpp32f)aCoeffs[1][0];
        perspective[4] = (Rpp32f)aCoeffs[1][1];
        perspective[5] = (Rpp32f)aCoeffs[1][2];
        perspective[6] = (Rpp32f)aCoeffs[2][0];
        perspective[7] = (Rpp32f)aCoeffs[2][1];
        perspective[8] = (Rpp32f)aCoeffs[2][2];

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        status = rppi_warp_perspective_u8_pln1_batchPD_gpu((RppPtr_t)pSrc, srcSize, maxSize, (RppPtr_t)pDst, dstSize, maxDstSize, perspective, noOfImages, handle);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        free(srcSize);
        free(dstSize);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiWarpPerspective_8u_C3R_Ctx(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp8u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3], int eInterpolation, NppStreamContext nppStreamCtx)
{
        int noOfImages = 1;
        int ip_channel = 3;//pkd_3
        RppiSize *srcSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize));
        RppiSize *dstSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize));
        RppiSize maxSize,maxDstSize;
        srcSize->width  = oSrcSize.width;
        srcSize->height = oSrcSize.height;
        dstSize->width  = oSrcSize.width;
        dstSize->height = oSrcSize.height;
        maxSize.width  = oSrcROI.width;
        maxSize.height = oSrcROI.height;
        maxDstSize.width = oSrcROI.width;
        maxDstSize.height = oSrcROI.height;
        RppStatus status;

        //Rpp32f perspective[9] = {aCoeffs[0][0],aCoeffs[0][1],aCoeffs[0][2],aCoeffs[1][0],aCoeffs[1][1],
        //aCoeffs[1][2],aCoeffs[2][0],aCoeffs[2][1],aCoeffs[2][2]};
        Rpp32f perspective[9] = {0};

        perspective[0] = (Rpp32f)aCoeffs[0][0];
        perspective[1] = (Rpp32f)aCoeffs[0][1];
        perspective[2] = (Rpp32f)aCoeffs[0][2];
        perspective[3] = (Rpp32f)aCoeffs[1][0];
        perspective[4] = (Rpp32f)aCoeffs[1][1];
        perspective[5] = (Rpp32f)aCoeffs[1][2];
        perspective[6] = (Rpp32f)aCoeffs[2][0];
        perspective[7] = (Rpp32f)aCoeffs[2][1];
        perspective[8] = (Rpp32f)aCoeffs[2][2];

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        status = rppi_warp_perspective_u8_pkd3_batchPD_gpu((RppPtr_t)pSrc, srcSize, maxSize, (RppPtr_t)pDst, dstSize, maxDstSize, perspective, noOfImages, handle);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        free(srcSize);
        free(dstSize);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiWarpPerspective_8u_P3R_Ctx(const Npp8u *pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp8u *pDst[3], int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3], int eInterpolation, NppStreamContext nppStreamCtx)
{
        int noOfImages = 1;
        int ip_channel = 3;//pln3
        RppiSize *srcSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize));
        RppiSize *dstSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize));
        RppiSize maxSize,maxDstSize;
        srcSize->width  = oSrcSize.width;
        srcSize->height = oSrcSize.height;
        dstSize->width  = oSrcSize.width;
        dstSize->height = oSrcSize.height;
        maxSize.width  = oSrcROI.width;
        maxSize.height = oSrcROI.height;
        maxDstSize.width = oSrcROI.width;
        maxDstSize.height = oSrcROI.height;
        RppStatus status;

        //Rpp32f perspective[9] = {aCoeffs[0][0],aCoeffs[0][1],aCoeffs[0][2],aCoeffs[1][0],aCoeffs[1][1],
        //aCoeffs[1][2],aCoeffs[2][0],aCoeffs[2][1],aCoeffs[2][2]};
        Rpp32f perspective[9] = {0};

        perspective[0] = (Rpp32f)aCoeffs[0][0];
        perspective[1] = (Rpp32f)aCoeffs[0][1];
        perspective[2] = (Rpp32f)aCoeffs[0][2];
        perspective[3] = (Rpp32f)aCoeffs[1][0];
        perspective[4] = (Rpp32f)aCoeffs[1][1];
        perspective[5] = (Rpp32f)aCoeffs[1][2];
        perspective[6] = (Rpp32f)aCoeffs[2][0];
        perspective[7] = (Rpp32f)aCoeffs[2][1];
        perspective[8] = (Rpp32f)aCoeffs[2][2];
		
	    Rpp8u *temp_in,*temp_output;
        hipMalloc(&temp_in, oSrcSize.height * oSrcSize.width * ip_channel);
	    hipMalloc(&temp_output, oSrcSize.height * oSrcSize.width * ip_channel);
        for (int i = 0; i < ip_channel; i++)
        {
            hipMemcpy(temp_in + (i * oSrcSize.height * oSrcSize.width), pSrc[i], oSrcSize.height * oSrcSize.width, hipMemcpyDeviceToDevice);
        }

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        status = rppi_warp_perspective_u8_pln3_batchPD_gpu((RppPtr_t)temp_in, srcSize, maxSize, (RppPtr_t)temp_output, dstSize, maxDstSize, perspective, noOfImages, handle);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
		
	    for (int m = 0; m < ip_channel; m++)
        {
            hipMemcpy(pDst[m],temp_output + (m * oSrcSize.height * oSrcSize.width), oSrcSize.height * oSrcSize.width, hipMemcpyDeviceToDevice);
        }
		
        free(srcSize);
        free(dstSize);
	    hipFree(temp_in);
        hipFree(temp_output);
	
        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiWarpPerspective_8u_P3R(const Npp8u *pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp8u *pDst[3], int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3], int eInterpolation)
{
        int noOfImages = 1;
        int ip_channel = 3;//pln3
        RppiSize *srcSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize));
        RppiSize *dstSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize));
        RppiSize maxSize,maxDstSize;
        srcSize->width  = oSrcSize.width;
        srcSize->height = oSrcSize.height;
        dstSize->width  = oSrcSize.width;
        dstSize->height = oSrcSize.height;
        maxSize.width  = oSrcROI.width;
        maxSize.height = oSrcROI.height;
        maxDstSize.width = oSrcROI.width;
        maxDstSize.height = oSrcROI.height;
        RppStatus status;

        //Rpp32f perspective[9] = {aCoeffs[0][0],aCoeffs[0][1],aCoeffs[0][2],aCoeffs[1][0],aCoeffs[1][1],
        //aCoeffs[1][2],aCoeffs[2][0],aCoeffs[2][1],aCoeffs[2][2]};
        Rpp32f perspective[9] = {0};

        perspective[0] = (Rpp32f)aCoeffs[0][0];
        perspective[1] = (Rpp32f)aCoeffs[0][1];
        perspective[2] = (Rpp32f)aCoeffs[0][2];
        perspective[3] = (Rpp32f)aCoeffs[1][0];
        perspective[4] = (Rpp32f)aCoeffs[1][1];
        perspective[5] = (Rpp32f)aCoeffs[1][2];
        perspective[6] = (Rpp32f)aCoeffs[2][0];
        perspective[7] = (Rpp32f)aCoeffs[2][1];
        perspective[8] = (Rpp32f)aCoeffs[2][2];
		
	    Rpp8u *temp_in,*temp_output;
        hipMalloc(&temp_in, oSrcSize.height * oSrcSize.width * ip_channel);
	    hipMalloc(&temp_output, oSrcSize.height * oSrcSize.width * ip_channel);
        for (int i = 0; i < ip_channel; i++)
        {
            hipMemcpy(temp_in + (i * oSrcSize.height * oSrcSize.width), pSrc[i], oSrcSize.height * oSrcSize.width, hipMemcpyDeviceToDevice);
        }

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        status = rppi_warp_perspective_u8_pln3_batchPD_gpu((RppPtr_t)temp_in, srcSize, maxSize, (RppPtr_t)temp_output, dstSize, maxDstSize, perspective, noOfImages, handle);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
		
	    for (int m = 0; m < ip_channel; m++)
        {
            hipMemcpy(pDst[m],temp_output + (m * oSrcSize.height * oSrcSize.width), oSrcSize.height * oSrcSize.width, hipMemcpyDeviceToDevice);
        }
		
        free(srcSize);
        free(dstSize);
	    hipFree(temp_in);
        hipFree(temp_output);
	
        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiMirror_8u_C3R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oROI, NppiAxis flip)
{
        int noOfImages = 1;
        int ip_channel = 3;//pkd_3
        RppiSize *srcSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize));
        RppiSize maxSize;
        srcSize->width  = oROI.width;
        srcSize->height = oROI.height;
        maxSize.width  = oROI.width;
        maxSize.height = oROI.height;
        RppStatus status;

	    Rpp32u flipAxis;

	    if(flip == NPP_HORIZONTAL_AXIS)
        {
            flipAxis = 0;
        }else if(flip == NPP_VERTICAL_AXIS)
        {
            flipAxis = 1;
        }else if(flip == NPP_BOTH_AXIS)
        {
            flipAxis = 2;
        }

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        status = rppi_flip_u8_pkd3_batchPD_gpu((RppPtr_t)pSrc, srcSize, maxSize, (RppPtr_t)pDst, &flipAxis, noOfImages, handle);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        free(srcSize);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiMirror_8u_C4R(const Npp8u *pSrc, int nSrcStep,
                            Npp8u *pDst, int nDstStep,
                            NppiSize oROI, NppiAxis flip)
{
    const int noOfImages = 1;
    const int ip_channel  = 4; //4-CHANNEL
    RppiSize *srcSize = (RppiSize*)calloc(noOfImages, sizeof(RppiSize));
    srcSize->width  = oROI.width;
    srcSize->height = oROI.height;
    RppiSize maxSize = *srcSize;

    Rpp32u flipAxis = 0;
    if      (flip == NPP_HORIZONTAL_AXIS) flipAxis = 0;
    else if (flip == NPP_VERTICAL_AXIS  ) flipAxis = 1;
    else if (flip == NPP_BOTH_AXIS      ) flipAxis = 2;

    rppHandle_t handle;
    hipStream_t stream;
    hipStreamCreate(&stream);
    rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

    RppStatus status = rppi_flip_u8_pkd4_batchPD_gpu(
        (RppPtr_t)pSrc, srcSize, maxSize, (RppPtr_t)pDst,
        &flipAxis, noOfImages, handle);

    hipDeviceSynchronize();
    rppDestroyGPU(handle);
    free(srcSize);

    return hipRppStatusTocudaNppStatus(status);
}

NppStatus nppiMirror_8u_C4IR(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip) {
    const int noOfImages = 1;
    const int ip_channel  = 4; //4-CHANNEL
    RppiSize *srcSize = (RppiSize*)calloc(noOfImages, sizeof(RppiSize));
    srcSize->width  = oROI.width;
    srcSize->height = oROI.height;
    RppiSize maxSize = *srcSize;

    int *pDst;
    hipMalloc(&pDst, oROI.width * oROI.height * ip_channel * sizeof(Rpp8u));

    Rpp32u flipAxis = 0;
    if      (flip == NPP_HORIZONTAL_AXIS) flipAxis = 0;
    else if (flip == NPP_VERTICAL_AXIS  ) flipAxis = 1;
    else if (flip == NPP_BOTH_AXIS      ) flipAxis = 2;

    rppHandle_t handle;
    hipStream_t stream;
    hipStreamCreate(&stream);
    rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

    RppStatus status = rppi_flip_u8_pkd4_batchPD_gpu(
        (RppPtr_t)pSrcDst, srcSize, maxSize, (RppPtr_t)pDst,
        &flipAxis, noOfImages, handle);

    hipDeviceSynchronize();
    hipMemcpy(pSrcDst, pDst, oROI.width * oROI.height * ip_channel * sizeof(Rpp8u), hipMemcpyDeviceToDevice);
    rppDestroyGPU(handle);
    free(srcSize);
    hipFree(pDst);

    return hipRppStatusTocudaNppStatus(status);
}

NppStatus nppiMirror_8u_C1R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oROI, NppiAxis flip, NppStreamContext nppStreamCtx)
{
        int noOfImages = 1;
        int ip_channel = 1;//pln1
        RppiSize *srcSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize));
        RppiSize maxSize;
        srcSize->width  = oROI.width;
        srcSize->height = oROI.height;
        maxSize.width  = oROI.width;
        maxSize.height = oROI.height;
        RppStatus status;

        Rpp32u flipAxis;

        if(flip == NPP_HORIZONTAL_AXIS)
        {
            flipAxis = 0;
        }else if(flip == NPP_VERTICAL_AXIS)
        {
            flipAxis = 1;
        }else if(flip == NPP_BOTH_AXIS)
        {
            flipAxis = 2;
        }

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        status = rppi_flip_u8_pln1_batchPD_gpu((RppPtr_t)pSrc, srcSize, maxSize, (RppPtr_t)pDst, &flipAxis, noOfImages, handle);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        free(srcSize);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiMirror_8u_C1R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oROI, NppiAxis flip)
{
        int noOfImages = 1;
        int ip_channel = 1;//pln1
        RppiSize *srcSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize));
        RppiSize maxSize;
        srcSize->width  = oROI.width; 
        srcSize->height = oROI.height; 
        maxSize.width  = oROI.width;
        maxSize.height = oROI.height;
        RppStatus status;

        Rpp32u flipAxis;

        if(flip == NPP_HORIZONTAL_AXIS)
        {
            flipAxis = 0;
        }else if(flip == NPP_VERTICAL_AXIS)
        {
            flipAxis = 1;
        }else if(flip == NPP_BOTH_AXIS)
        {
            flipAxis = 2;
        }

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        status = rppi_flip_u8_pln1_batchPD_gpu((RppPtr_t)pSrc, srcSize, maxSize, (RppPtr_t)pDst, &flipAxis, noOfImages, handle);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        free(srcSize);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiMirror_8u_C1IR_Ctx(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip, NppStreamContext nppStreamCtx)
{
        int noOfImages = 1;
        int ip_channel = 1;//pln1
        RppiSize *srcSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize));
        RppiSize maxSize;
        srcSize->width  = oROI.width;
        srcSize->height = oROI.height;
        maxSize.width  = oROI.width;
        maxSize.height = oROI.height;
	    int *pDst;
	    hipMalloc(&pDst, oROI.width * oROI.height * sizeof(Rpp8u));
        RppStatus status;

        Rpp32u flipAxis;

        if(flip == NPP_HORIZONTAL_AXIS)
        {
            flipAxis = 0;
        }else if(flip == NPP_VERTICAL_AXIS)
        {
            flipAxis = 1;
        }else if(flip == NPP_BOTH_AXIS)
        {
            flipAxis = 2;
        }

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        status = rppi_flip_u8_pln1_batchPD_gpu((RppPtr_t)pSrcDst, srcSize, maxSize, (RppPtr_t)pDst, &flipAxis, noOfImages, handle);
        hipDeviceSynchronize();
	    hipMemcpy(pSrcDst, pDst, oROI.width * oROI.height * sizeof(Rpp8u), hipMemcpyDeviceToDevice);

        rppDestroyGPU(handle);
        free(srcSize);
	    hipFree(pDst);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiMirror_8u_C1IR(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip)
{
        int noOfImages = 1;
        int ip_channel = 1;//pln1
        RppiSize *srcSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize)); 
        RppiSize maxSize;
        srcSize->width  = oROI.width;
        srcSize->height = oROI.height;
        maxSize.width  = oROI.width;
        maxSize.height = oROI.height;
        int *pDst;
        hipMalloc(&pDst, oROI.width * oROI.height * sizeof(Rpp8u));
        RppStatus status;

        Rpp32u flipAxis;

        if(flip == NPP_HORIZONTAL_AXIS)
        {
            flipAxis = 0;
        }else if(flip == NPP_VERTICAL_AXIS)
        {
            flipAxis = 1;
        }else if(flip == NPP_BOTH_AXIS)
        {
            flipAxis = 2;
        }

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        status = rppi_flip_u8_pln1_batchPD_gpu((RppPtr_t)pSrcDst, srcSize, maxSize, (RppPtr_t)pDst, &flipAxis, noOfImages, handle);
        hipDeviceSynchronize();
        hipMemcpy(pSrcDst, pDst, oROI.width * oROI.height * sizeof(Rpp8u), hipMemcpyDeviceToDevice);

        rppDestroyGPU(handle);
        free(srcSize);
        hipFree(pDst);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiMirror_8u_C3R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oROI, NppiAxis flip, NppStreamContext nppStreamCtx)
{
        int noOfImages = 1;
        int ip_channel = 3;//pkd_3
        RppiSize *srcSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize));
        RppiSize maxSize;
        srcSize->width  = oROI.width;
        srcSize->height = oROI.height;
        maxSize.width  = oROI.width;
        maxSize.height = oROI.height;
        RppStatus status;

        Rpp32u flipAxis;

        if(flip == NPP_HORIZONTAL_AXIS)
        {
            flipAxis = 0;
        }else if(flip == NPP_VERTICAL_AXIS)
        {
            flipAxis = 1;
        }else if(flip == NPP_BOTH_AXIS)
        {
            flipAxis = 2;
        }

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        status = rppi_flip_u8_pkd3_batchPD_gpu((RppPtr_t)pSrc, srcSize, maxSize, (RppPtr_t)pDst, &flipAxis, noOfImages, handle);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        free(srcSize);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiMirror_8u_C3IR_Ctx(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip, NppStreamContext nppStreamCtx)
{
        int noOfImages = 1;
        int ip_channel = 3;//pkd3
        RppiSize *srcSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize)); 
        RppiSize maxSize;
        srcSize->width  = oROI.width;
        srcSize->height = oROI.height;
        maxSize.width  = oROI.width;
        maxSize.height = oROI.height;
        int *pDst;
        hipMalloc(&pDst, oROI.width * oROI.height * ip_channel * sizeof(Rpp8u));
        RppStatus status;

        Rpp32u flipAxis;

        if(flip == NPP_HORIZONTAL_AXIS)
        {
            flipAxis = 0;
        }else if(flip == NPP_VERTICAL_AXIS)
        {
            flipAxis = 1;
        }else if(flip == NPP_BOTH_AXIS)
        {
            flipAxis = 2;
        }

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        status = rppi_flip_u8_pkd3_batchPD_gpu((RppPtr_t)pSrcDst, srcSize, maxSize, (RppPtr_t)pDst, &flipAxis, noOfImages, handle);
        hipDeviceSynchronize();
        hipMemcpy(pSrcDst, pDst, oROI.width * oROI.height * ip_channel * sizeof(Rpp8u), hipMemcpyDeviceToDevice);

        rppDestroyGPU(handle);
        free(srcSize);
        hipFree(pDst);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiMirror_8u_C3IR(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip)
{
        int noOfImages = 1;
        int ip_channel = 3;//pkd3
        RppiSize *srcSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize));
        RppiSize maxSize;
        srcSize->width  = oROI.width;
        srcSize->height = oROI.height;
        maxSize.width  = oROI.width;
        maxSize.height = oROI.height;
        int *pDst;
        hipMalloc(&pDst, oROI.width * oROI.height * ip_channel * sizeof(Rpp8u));
        RppStatus status;

        Rpp32u flipAxis;

        if(flip == NPP_HORIZONTAL_AXIS)
        {
            flipAxis = 0;
        }else if(flip == NPP_VERTICAL_AXIS)
        {
            flipAxis = 1;
        }else if(flip == NPP_BOTH_AXIS)
        {
            flipAxis = 2;
        }

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        status = rppi_flip_u8_pkd3_batchPD_gpu((RppPtr_t)pSrcDst, srcSize, maxSize, (RppPtr_t)pDst, &flipAxis, noOfImages, handle);
        hipDeviceSynchronize();
        hipMemcpy(pSrcDst, pDst, oROI.width * oROI.height * ip_channel * sizeof(Rpp8u), hipMemcpyDeviceToDevice);

        rppDestroyGPU(handle);
        free(srcSize);
        hipFree(pDst);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiMirror_32s_C4R(const Npp32s *pSrc, int nSrcStep, Npp32s *pDst, int nDstStep, NppiSize oROI, NppiAxis flip) 
{
    const int noOfImages = 1;
    const int ip_channel  = 4; //4-CHANNEL
    RppiSize *srcSize = (RppiSize*)calloc(noOfImages, sizeof(RppiSize));
    srcSize->width  = oROI.width;
    srcSize->height = oROI.height;
    RppiSize maxSize = *srcSize;

    Rpp32u flipAxis = 0;
    if      (flip == NPP_HORIZONTAL_AXIS) flipAxis = 0;
    else if (flip == NPP_VERTICAL_AXIS  ) flipAxis = 1;
    else if (flip == NPP_BOTH_AXIS      ) flipAxis = 2;

    rppHandle_t handle;
    hipStream_t stream;
    hipStreamCreate(&stream);
    rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

    RpptDataType dataType = S32;

    RppStatus status = rppi_flip_pkd4_batchPD_gpu(
        (RppPtr_t)pSrc, dataType, srcSize, maxSize, (RppPtr_t)pDst,
        &flipAxis, noOfImages, handle);

    hipDeviceSynchronize();
    rppDestroyGPU(handle);
    free(srcSize);

    return hipRppStatusTocudaNppStatus(status);
}

NppStatus nppiMirror_32f_C4R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oROI, NppiAxis flip)
{
    const int noOfImages = 1;
    const int ip_channel  = 4; //4-CHANNEL
    RppiSize *srcSize = (RppiSize*)calloc(noOfImages, sizeof(RppiSize));
    srcSize->width  = oROI.width;
    srcSize->height = oROI.height;
    RppiSize maxSize = *srcSize;

    Rpp32u flipAxis = 0;
    if      (flip == NPP_HORIZONTAL_AXIS) flipAxis = 0;
    else if (flip == NPP_VERTICAL_AXIS  ) flipAxis = 1;
    else if (flip == NPP_BOTH_AXIS      ) flipAxis = 2;

    rppHandle_t handle;
    hipStream_t stream;
    hipStreamCreate(&stream);
    rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

    RpptDataType dataType = F32;

    RppStatus status = rppi_flip_pkd4_batchPD_gpu(
        (RppPtr_t)pSrc, dataType, srcSize, maxSize, (RppPtr_t)pDst,
        &flipAxis, noOfImages, handle);

    hipDeviceSynchronize();
    rppDestroyGPU(handle);
    free(srcSize);

    return hipRppStatusTocudaNppStatus(status);
}

NppStatus nppiMirror_16u_C4R(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep, NppiSize oROI, NppiAxis flip)
{
    const int noOfImages = 1;
    const int ip_channel  = 4; //4-CHANNEL
    RppiSize *srcSize = (RppiSize*)calloc(noOfImages, sizeof(RppiSize));
    srcSize->width  = oROI.width;
    srcSize->height = oROI.height;
    RppiSize maxSize = *srcSize;

    Rpp32u flipAxis = 0;
    if      (flip == NPP_HORIZONTAL_AXIS) flipAxis = 0;
    else if (flip == NPP_VERTICAL_AXIS  ) flipAxis = 1;
    else if (flip == NPP_BOTH_AXIS      ) flipAxis = 2;

    rppHandle_t handle;
    hipStream_t stream;
    hipStreamCreate(&stream);
    rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

    RpptDataType dataType = U16;

    RppStatus status = rppi_flip_pkd4_batchPD_gpu(
        (RppPtr_t)pSrc, dataType, srcSize, maxSize, (RppPtr_t)pDst,
        &flipAxis, noOfImages, handle);

    hipDeviceSynchronize();
    rppDestroyGPU(handle);
    free(srcSize);

    return hipRppStatusTocudaNppStatus(status);
}

NppStatus nppiMirror_32f_C4IR(Npp32f *pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip)
{
    const int noOfImages = 1;
    const int ip_channel  = 4; //4-CHANNEL
    RppiSize *srcSize = (RppiSize*)calloc(noOfImages, sizeof(RppiSize));
    srcSize->width  = oROI.width;
    srcSize->height = oROI.height;
    RppiSize maxSize = *srcSize;

    int *pDst;
    hipMalloc(&pDst, oROI.width * oROI.height * ip_channel * sizeof(Rpp32f));

    Rpp32u flipAxis = 0;
    if      (flip == NPP_HORIZONTAL_AXIS) flipAxis = 0;
    else if (flip == NPP_VERTICAL_AXIS  ) flipAxis = 1;
    else if (flip == NPP_BOTH_AXIS      ) flipAxis = 2;

    rppHandle_t handle;
    hipStream_t stream;
    hipStreamCreate(&stream);
    rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

    RpptDataType dataType = F32;

    RppStatus status = rppi_flip_pkd4_batchPD_gpu(
        (RppPtr_t)pSrcDst, dataType, srcSize, maxSize, (RppPtr_t)pDst,
        &flipAxis, noOfImages, handle);

    hipDeviceSynchronize();
    hipMemcpy(pSrcDst, pDst, oROI.width * oROI.height * ip_channel * sizeof(Rpp32f), hipMemcpyDeviceToDevice);
    rppDestroyGPU(handle);
    free(srcSize);
    hipFree(pDst);

    return hipRppStatusTocudaNppStatus(status);
}

NppStatus nppiMirror_32s_C4IR(Npp32s *pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip)
{
    const int noOfImages = 1;
    const int ip_channel  = 4; //4-CHANNEL
    RppiSize *srcSize = (RppiSize*)calloc(noOfImages, sizeof(RppiSize));
    srcSize->width  = oROI.width;
    srcSize->height = oROI.height;
    RppiSize maxSize = *srcSize;

    int *pDst;
    hipMalloc(&pDst, oROI.width * oROI.height * ip_channel * sizeof(Rpp32s));

    Rpp32u flipAxis = 0;
    if      (flip == NPP_HORIZONTAL_AXIS) flipAxis = 0;
    else if (flip == NPP_VERTICAL_AXIS  ) flipAxis = 1;
    else if (flip == NPP_BOTH_AXIS      ) flipAxis = 2;

    rppHandle_t handle;
    hipStream_t stream;
    hipStreamCreate(&stream);
    rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

    RpptDataType dataType = S32;

    RppStatus status = rppi_flip_pkd4_batchPD_gpu(
        (RppPtr_t)pSrcDst, dataType, srcSize, maxSize, (RppPtr_t)pDst,
        &flipAxis, noOfImages, handle);

    hipDeviceSynchronize();
    hipMemcpy(pSrcDst, pDst, oROI.width * oROI.height * ip_channel * sizeof(Rpp32s), hipMemcpyDeviceToDevice);
    rppDestroyGPU(handle);
    free(srcSize);
    hipFree(pDst);

    return hipRppStatusTocudaNppStatus(status);
}

NppStatus nppiMirror_16u_C4IR(Npp16u *pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip)
{
    const int noOfImages = 1;
    const int ip_channel  = 4; //4-CHANNEL
    RppiSize *srcSize = (RppiSize*)calloc(noOfImages, sizeof(RppiSize));
    srcSize->width  = oROI.width;
    srcSize->height = oROI.height;
    RppiSize maxSize = *srcSize;

    int *pDst;
    hipMalloc(&pDst, oROI.width * oROI.height * ip_channel * sizeof(Rpp16u));

    Rpp32u flipAxis = 0;
    if      (flip == NPP_HORIZONTAL_AXIS) flipAxis = 0;
    else if (flip == NPP_VERTICAL_AXIS  ) flipAxis = 1;
    else if (flip == NPP_BOTH_AXIS      ) flipAxis = 2;

    rppHandle_t handle;
    hipStream_t stream;
    hipStreamCreate(&stream);
    rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

    RpptDataType dataType = U16;

    RppStatus status = rppi_flip_pkd4_batchPD_gpu(
        (RppPtr_t)pSrcDst, dataType, srcSize, maxSize, (RppPtr_t)pDst,
        &flipAxis, noOfImages, handle);

    hipDeviceSynchronize();
    hipMemcpy(pSrcDst, pDst, oROI.width * oROI.height * ip_channel * sizeof(Rpp16u), hipMemcpyDeviceToDevice);
    rppDestroyGPU(handle);
    free(srcSize);
    hipFree(pDst);

    return hipRppStatusTocudaNppStatus(status);
}

NppStatus nppiMirror_32s_C3R(const Npp32s *pSrc, int nSrcStep, Npp32s *pDst, int nDstStep, NppiSize oROI, NppiAxis flip)
{
    int noOfImages = 1;
    int ip_channel = 3;//pkd_3
    RppiSize *srcSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize));
    RppiSize maxSize;
    srcSize->width  = oROI.width;
    srcSize->height = oROI.height;
    maxSize.width  = oROI.width;
    maxSize.height = oROI.height;
    RppStatus status;

	Rpp32u flipAxis;

	if(flip == NPP_HORIZONTAL_AXIS)
    {
        flipAxis = 0;
    }else if(flip == NPP_VERTICAL_AXIS)
    {
        flipAxis = 1;
    }else if(flip == NPP_BOTH_AXIS)
    {
        flipAxis = 2;
    }

    rppHandle_t handle;
    hipStream_t stream;
    hipStreamCreate(&stream);
    rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

    RpptDataType dataType = S32;

    status = rppi_flip_pkd3_batchPD_gpu((RppPtr_t)pSrc, dataType, srcSize, maxSize, (RppPtr_t)pDst, &flipAxis, noOfImages, handle);
    hipDeviceSynchronize();

    rppDestroyGPU(handle);
    free(srcSize);

    return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiMirror_32f_C3R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oROI, NppiAxis flip)
{
    int noOfImages = 1;
    int ip_channel = 3;//pkd_3
    RppiSize *srcSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize));
    RppiSize maxSize;
    srcSize->width  = oROI.width;
    srcSize->height = oROI.height;
    maxSize.width  = oROI.width;
    maxSize.height = oROI.height;
    RppStatus status;

	Rpp32u flipAxis;

	if(flip == NPP_HORIZONTAL_AXIS)
    {
        flipAxis = 0;
    }else if(flip == NPP_VERTICAL_AXIS)
    {
        flipAxis = 1;
    }else if(flip == NPP_BOTH_AXIS)
    {
        flipAxis = 2;
    }

    rppHandle_t handle;
    hipStream_t stream;
    hipStreamCreate(&stream);
    rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

    RpptDataType dataType = F32;

    status = rppi_flip_pkd3_batchPD_gpu((RppPtr_t)pSrc, dataType, srcSize, maxSize, (RppPtr_t)pDst, &flipAxis, noOfImages, handle);
    hipDeviceSynchronize();

    rppDestroyGPU(handle);
    free(srcSize);

    return(hipRppStatusTocudaNppStatus(status));    
}

NppStatus nppiMirror_32s_C3IR(Npp32s *pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip)
{
    int noOfImages = 1;
    int ip_channel = 3;//pkd3
    RppiSize *srcSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize));
    RppiSize maxSize;
    srcSize->width  = oROI.width;
    srcSize->height = oROI.height;
    maxSize.width  = oROI.width;
    maxSize.height = oROI.height;
    int *pDst;
    hipMalloc(&pDst, oROI.width * oROI.height * ip_channel * sizeof(Rpp32s));
    RppStatus status;

    Rpp32u flipAxis;

    if(flip == NPP_HORIZONTAL_AXIS)
    {
        flipAxis = 0;
    }else if(flip == NPP_VERTICAL_AXIS)
    {
        flipAxis = 1;
    }else if(flip == NPP_BOTH_AXIS)
    {
        flipAxis = 2;
    }

    rppHandle_t handle;
    hipStream_t stream;
    hipStreamCreate(&stream);
    rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

    RpptDataType dataType = S32;

    status = rppi_flip_pkd3_batchPD_gpu((RppPtr_t)pSrcDst, dataType, srcSize, maxSize, (RppPtr_t)pDst, &flipAxis, noOfImages, handle);
    hipDeviceSynchronize();
    hipMemcpy(pSrcDst, pDst, oROI.width * oROI.height * ip_channel * sizeof(Rpp32s), hipMemcpyDeviceToDevice);

    rppDestroyGPU(handle);
    free(srcSize);
    hipFree(pDst);

    return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiMirror_32f_C3IR(Npp32f *pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip)
{
    int noOfImages = 1;
    int ip_channel = 3;//pkd3
    RppiSize *srcSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize));
    RppiSize maxSize;
    srcSize->width  = oROI.width;
    srcSize->height = oROI.height;
    maxSize.width  = oROI.width;
    maxSize.height = oROI.height;
    int *pDst;
    hipMalloc(&pDst, oROI.width * oROI.height * ip_channel * sizeof(Rpp32f));
    RppStatus status;

    Rpp32u flipAxis;

    if(flip == NPP_HORIZONTAL_AXIS)
    {
        flipAxis = 0;
    }else if(flip == NPP_VERTICAL_AXIS)
    {
        flipAxis = 1;
    }else if(flip == NPP_BOTH_AXIS)
    {
        flipAxis = 2;
    }

    rppHandle_t handle;
    hipStream_t stream;
    hipStreamCreate(&stream);
    rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

    RpptDataType dataType = F32;

    status = rppi_flip_pkd3_batchPD_gpu((RppPtr_t)pSrcDst, dataType, srcSize, maxSize, (RppPtr_t)pDst, &flipAxis, noOfImages, handle);
    hipDeviceSynchronize();
    hipMemcpy(pSrcDst, pDst, oROI.width * oROI.height * ip_channel * sizeof(Rpp32f), hipMemcpyDeviceToDevice);

    rppDestroyGPU(handle);
    free(srcSize);
    hipFree(pDst);

    return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiMirror_16u_C3IR(Npp16u *pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip)
{
    int noOfImages = 1;
    int ip_channel = 3;//pkd3
    RppiSize *srcSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize));
    RppiSize maxSize;
    srcSize->width  = oROI.width;
    srcSize->height = oROI.height;
    maxSize.width  = oROI.width;
    maxSize.height = oROI.height;
    int *pDst;
    hipMalloc(&pDst, oROI.width * oROI.height * ip_channel * sizeof(Rpp16u));
    RppStatus status;

    Rpp32u flipAxis;

    if(flip == NPP_HORIZONTAL_AXIS)
    {
        flipAxis = 0;
    }else if(flip == NPP_VERTICAL_AXIS)
    {
        flipAxis = 1;
    }else if(flip == NPP_BOTH_AXIS)
    {
        flipAxis = 2;
    }

    rppHandle_t handle;
    hipStream_t stream;
    hipStreamCreate(&stream);
    rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

    RpptDataType dataType = U16;

    status = rppi_flip_pkd3_batchPD_gpu((RppPtr_t)pSrcDst, dataType, srcSize, maxSize, (RppPtr_t)pDst, &flipAxis, noOfImages, handle);
    hipDeviceSynchronize();
    hipMemcpy(pSrcDst, pDst, oROI.width * oROI.height * ip_channel * sizeof(Rpp16u), hipMemcpyDeviceToDevice);

    rppDestroyGPU(handle);
    free(srcSize);
    hipFree(pDst);

    return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiMirror_32s_C1R(const Npp32s *pSrc, int nSrcStep, Npp32s *pDst, int nDstStep, NppiSize oROI, NppiAxis flip)
{
    int noOfImages = 1;
    int ip_channel = 1;//pln1
    RppiSize *srcSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize));
    RppiSize maxSize;
    srcSize->width  = oROI.width; 
    srcSize->height = oROI.height; 
    maxSize.width  = oROI.width;
    maxSize.height = oROI.height;
    RppStatus status;

    Rpp32u flipAxis;

    if(flip == NPP_HORIZONTAL_AXIS)
    {
        flipAxis = 0;
    }else if(flip == NPP_VERTICAL_AXIS)
    {
        flipAxis = 1;
    }else if(flip == NPP_BOTH_AXIS)
    {
        flipAxis = 2;
    }

    rppHandle_t handle;
    hipStream_t stream;
    hipStreamCreate(&stream);
    rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

    RpptDataType dataType = S32;

    status = rppi_flip_pln1_batchPD_gpu((RppPtr_t)pSrc, dataType, srcSize, maxSize, (RppPtr_t)pDst, &flipAxis, noOfImages, handle);
    hipDeviceSynchronize();

    rppDestroyGPU(handle);
    free(srcSize);

    return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiMirror_32f_C1R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oROI, NppiAxis flip)
{
    int noOfImages = 1;
    int ip_channel = 1;//pln1
    RppiSize *srcSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize));
    RppiSize maxSize;
    srcSize->width  = oROI.width; 
    srcSize->height = oROI.height; 
    maxSize.width  = oROI.width;
    maxSize.height = oROI.height;
    RppStatus status;

    Rpp32u flipAxis;

    if(flip == NPP_HORIZONTAL_AXIS)
    {
        flipAxis = 0;
    }else if(flip == NPP_VERTICAL_AXIS)
    {
        flipAxis = 1;
    }else if(flip == NPP_BOTH_AXIS)
    {
        flipAxis = 2;
    }

    rppHandle_t handle;
    hipStream_t stream;
    hipStreamCreate(&stream);
    rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

    RpptDataType dataType = F32;

    status = rppi_flip_pln1_batchPD_gpu((RppPtr_t)pSrc, dataType, srcSize, maxSize, (RppPtr_t)pDst, &flipAxis, noOfImages, handle);
    hipDeviceSynchronize();

    rppDestroyGPU(handle);
    free(srcSize);

    return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiMirror_16u_C1R(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep, NppiSize oROI, NppiAxis flip)
{
    int noOfImages = 1;
    int ip_channel = 1;//pln1
    RppiSize *srcSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize));
    RppiSize maxSize;
    srcSize->width  = oROI.width; 
    srcSize->height = oROI.height; 
    maxSize.width  = oROI.width;
    maxSize.height = oROI.height;
    RppStatus status;

    Rpp32u flipAxis;

    if(flip == NPP_HORIZONTAL_AXIS)
    {
        flipAxis = 0;
    }else if(flip == NPP_VERTICAL_AXIS)
    {
        flipAxis = 1;
    }else if(flip == NPP_BOTH_AXIS)
    {
        flipAxis = 2;
    }

    rppHandle_t handle;
    hipStream_t stream;
    hipStreamCreate(&stream);
    rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

    RpptDataType dataType = U16;

    status = rppi_flip_pln1_batchPD_gpu((RppPtr_t)pSrc, dataType, srcSize, maxSize, (RppPtr_t)pDst, &flipAxis, noOfImages, handle);
    hipDeviceSynchronize();

    rppDestroyGPU(handle);
    free(srcSize);

    return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiMirror_32s_C1IR(Npp32s *pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip)
{
    int noOfImages = 1;
    int ip_channel = 1;//pln1
    RppiSize *srcSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize)); 
    RppiSize maxSize;
    srcSize->width  = oROI.width;
    srcSize->height = oROI.height;
    maxSize.width  = oROI.width;
    maxSize.height = oROI.height;
    int *pDst;
    hipMalloc(&pDst, oROI.width * oROI.height * sizeof(Rpp32s));
    RppStatus status;

    Rpp32u flipAxis;

    if(flip == NPP_HORIZONTAL_AXIS)
    {
        flipAxis = 0;
    }else if(flip == NPP_VERTICAL_AXIS)
    {
        flipAxis = 1;
    }else if(flip == NPP_BOTH_AXIS)
    {
        flipAxis = 2;
    }

    rppHandle_t handle;
    hipStream_t stream;
    hipStreamCreate(&stream);
    rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

    RpptDataType dataType = S32;
    status = rppi_flip_pln1_batchPD_gpu((RppPtr_t)pSrcDst, dataType, srcSize, maxSize, (RppPtr_t)pDst, &flipAxis, noOfImages, handle);
    hipDeviceSynchronize();
    hipMemcpy(pSrcDst, pDst, oROI.width * oROI.height * sizeof(Rpp32s), hipMemcpyDeviceToDevice);

    rppDestroyGPU(handle);
    free(srcSize);
    hipFree(pDst);

    return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiMirror_32f_C1IR(Npp32f *pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip)
{
    int noOfImages = 1;
    int ip_channel = 1;//pln1
    RppiSize *srcSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize)); 
    RppiSize maxSize;
    srcSize->width  = oROI.width;
    srcSize->height = oROI.height;
    maxSize.width  = oROI.width;
    maxSize.height = oROI.height;
    int *pDst;
    hipMalloc(&pDst, oROI.width * oROI.height * sizeof(Rpp32f));
    RppStatus status;

    Rpp32u flipAxis;

    if(flip == NPP_HORIZONTAL_AXIS)
    {
        flipAxis = 0;
    }else if(flip == NPP_VERTICAL_AXIS)
    {
        flipAxis = 1;
    }else if(flip == NPP_BOTH_AXIS)
    {
        flipAxis = 2;
    }

    rppHandle_t handle;
    hipStream_t stream;
    hipStreamCreate(&stream);
    rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

    RpptDataType dataType = F32;
    status = rppi_flip_pln1_batchPD_gpu((RppPtr_t)pSrcDst, dataType, srcSize, maxSize, (RppPtr_t)pDst, &flipAxis, noOfImages, handle);
    hipDeviceSynchronize();
    hipMemcpy(pSrcDst, pDst, oROI.width * oROI.height * sizeof(Rpp32f), hipMemcpyDeviceToDevice);

    rppDestroyGPU(handle);
    free(srcSize);
    hipFree(pDst);

    return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiMirror_16u_C1IR(Npp16u *pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip)
{
    int noOfImages = 1;
    int ip_channel = 1;//pln1
    RppiSize *srcSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize)); 
    RppiSize maxSize;
    srcSize->width  = oROI.width;
    srcSize->height = oROI.height;
    maxSize.width  = oROI.width;
    maxSize.height = oROI.height;
    int *pDst;
    hipMalloc(&pDst, oROI.width * oROI.height * sizeof(Rpp16u));
    RppStatus status;

    Rpp32u flipAxis;

    if(flip == NPP_HORIZONTAL_AXIS)
    {
        flipAxis = 0;
    }else if(flip == NPP_VERTICAL_AXIS)
    {
        flipAxis = 1;
    }else if(flip == NPP_BOTH_AXIS)
    {
        flipAxis = 2;
    }

    rppHandle_t handle;
    hipStream_t stream;
    hipStreamCreate(&stream);
    rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

    RpptDataType dataType = U16;
    status = rppi_flip_pln1_batchPD_gpu((RppPtr_t)pSrcDst, dataType, srcSize, maxSize, (RppPtr_t)pDst, &flipAxis, noOfImages, handle);
    hipDeviceSynchronize();
    hipMemcpy(pSrcDst, pDst, oROI.width * oROI.height * sizeof(Rpp16u), hipMemcpyDeviceToDevice);

    rppDestroyGPU(handle);
    free(srcSize);
    hipFree(pDst);

    return(hipRppStatusTocudaNppStatus(status));
}

/*NppStatus nppiWarpAffine_8u_P3R(const Npp8u *pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp8u *pDst[3], int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3], int eInterpolation)
{
        int noOfImages = 1;
        int ip_channel = 3;//pln3
        RppiSize *srcSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize));
        RppiSize *dstSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize));
        RppiSize maxSize,maxDstSize;
        srcSize->width  = oSrcSize.width;
        srcSize->height = oSrcSize.height;
        dstSize->width  = oSrcSize.width;
        dstSize->height = oSrcSize.height;
        maxSize.width  = oSrcROI.width;
        maxSize.height = oSrcROI.height;
        maxDstSize.width = oSrcROI.width;
        maxDstSize.height = oSrcROI.height;
	unsigned int outputFormatToggle = 0;
        RppStatus status;

        Rpp32f6 affineTensor_f6[1];
        Rpp32f *affineTensor = (Rpp32f *)affineTensor_f6;
        affineTensor_f6[0].data[0] = aCoeffs[0][0];
        affineTensor_f6[0].data[1] = aCoeffs[0][1];
        affineTensor_f6[0].data[2] = aCoeffs[0][2];
        affineTensor_f6[0].data[3] = aCoeffs[1][0];
        affineTensor_f6[0].data[4] = aCoeffs[1][1];
        affineTensor_f6[0].data[5] = aCoeffs[1][2];

        Rpp8u *temp_in,*temp_output;
        hipMalloc(&temp_in, oSrcSize.height * oSrcSize.width * ip_channel);
        hipMalloc(&temp_output, oSrcSize.height * oSrcSize.width * ip_channel);
        for (int i = 0; i < ip_channel; i++)
        {
            hipMemcpy(temp_in + (i * oSrcSize.height * oSrcSize.width), pSrc[i], oSrcSize.height * oSrcSize.width, hipMemcpyDeviceToDevice);
        }
		
	rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);
        clock_t start, end;
        double gpu_time_used;

        start = clock();
        //status = rppi_warp_perspective_u8_pln3_batchPD_gpu((RppPtr_t)temp_in, srcSize, maxSize, (RppPtr_t)temp_output, dstSize, maxDstSize, perspective, noOfImages, handle);
	status = rppi_warp_affine_u8_pln3_batchPD_gpu((RppPtr_t)temp_in, srcSize, maxSize, (RppPtr_t)temp_output, dstSize, maxDstSize, affineTensor, outputFormatToggle, noOfImages, handle);
        hipDeviceSynchronize();
        end = clock();

        printf("\nnppiWarpAffine_8u_P3R is %d\n", status);
        gpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
        std::cout << "\nGPU WarpAffine_8u -  : " << gpu_time_used;
        printf("\n");
        rppDestroyGPU(handle);

        for (int m = 0; m < ip_channel; m++)
        {
            hipMemcpy(pDst[m],temp_output + (m * oSrcSize.height * oSrcSize.width), oSrcSize.height * oSrcSize.width, hipMemcpyDeviceToDevice);
        }

        free(srcSize);
        free(dstSize);
        hipFree(temp_in);
        hipFree(temp_output);

        return(hipRppStatusTocudaNppStatus(status));
}*/

/*NppStatus nppiResize_8u_P3R(const Npp8u *pSrc[3], int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI, Npp8u *pDst[3], int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI, int eInterpolation)
{
        int noOfImages = 1;
        int ip_channel = 3;//pln3
        RppiSize *srcSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize));
        RppiSize *dstSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize));
        RppiSize maxSize,maxDstSize;
        srcSize->width  = oSrcSize.width;
        srcSize->height = oSrcSize.height;
        dstSize->width  = oDstSize.width;
        dstSize->height = oDstSize.height;
        maxSize.width  = oSrcRectROI.width;
        maxSize.height = oSrcRectROI.height;
        maxDstSize.width = oDstRectROI.width;
        maxDstSize.height = oDstRectROI.height;
	unsigned int outputFormatToggle = 0;
        RppStatus status;

        Rpp8u *temp_in,*temp_output;
        hipMalloc(&temp_in, oSrcSize.height * oSrcSize.width * ip_channel);
        hipMalloc(&temp_output, oDstSize.height * oDstSize.width * ip_channel);
        for (int i = 0; i < ip_channel; i++)
        {
            hipMemcpy(temp_in + (i * oSrcSize.height * oSrcSize.width), pSrc[i], oSrcSize.height * oSrcSize.width, hipMemcpyDeviceToDevice);
        }
		
	rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);
        clock_t start, end;
        double gpu_time_used;

        start = clock();
        //status = rppi_warp_perspective_u8_pln3_batchPD_gpu((RppPtr_t)temp_in, srcSize, maxSize, (RppPtr_t)temp_output, dstSize, maxDstSize, perspective, noOfImages, handle);
	status = rppi_resize_u8_pln3_batchPD_gpu((RppPtr_t)temp_in, srcSize, maxSize, (RppPtr_t)temp_output, dstSize, maxDstSize, outputFormatToggle, noOfImages, handle);
        hipDeviceSynchronize();
        end = clock();

        printf("\nnppiResize_8u_P3R is %d\n", status);
        gpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
        std::cout << "\nGPU Resize_8u -  : " << gpu_time_used;
        printf("\n");
        rppDestroyGPU(handle);

        for (int m = 0; m < ip_channel; m++)
        {
            hipMemcpy(pDst[m],temp_output + (m * oDstSize.height * oDstSize.width), oDstSize.height * oDstSize.width, hipMemcpyDeviceToDevice);
        }

        free(srcSize);
        free(dstSize);
        hipFree(temp_in);
        hipFree(temp_output);

        return(hipRppStatusTocudaNppStatus(status));
}*/

#endif // GPU_SUPPORT

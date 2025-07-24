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

#ifndef RPPI_FILTER_OPERATIONS_H
#define RPPI_FILTER_OPERATIONS_H

#include "rppdefs.h"
#include "rpp.h"
#ifdef __cplusplus
extern "C" {
#endif

/*!
 * \file
 * \brief RPPI Image Operations - Filter Operations - To be deprecated.
 * \defgroup group_rppi_filter_operations RPPI Image Operations - Filter Operations
 * \brief RPPI Image Operations - Filter Operations - To be deprecated.
 * \deprecated
 */

/*! \addtogroup group_rppi_filter_operations
 * @{
 */

/******************** box_filter ********************/

// Applies the box filter over every pixel using a [kernelSize X kernelSize] square mask for a batch of images

// *param[in] srcPtr Input image batch
// *param[in] srcSize Array containing an RppiSize for each image in the batch
// *param[in] maxSrcSize A single RppiSize which is the maxWidth and maxHeight for all images in the batch
// *param[out] dstPtr Output image batch
// *param[in] kernelSize Array containing an Rpp32u kernel size for each image in the batch (kernelSize[n] = 3/5/7 for optimal use)
// *param[in] nbatchSize Batch size or the number of images in the batch
// *param[in] rppHandle OpenCL-handle/HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : No error, Succesful completion
// *retval RPP_ERROR : Error

RppStatus rppi_box_filter_u8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *kernelSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_box_filter_u8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *kernelSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_box_filter_u8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *kernelSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
#ifdef GPU_SUPPORT
RppStatus rppi_box_filter_u8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *kernelSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_box_filter_u8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *kernelSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_box_filter_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *kernelSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/******************** sobel_filter ********************/

// Applies the sobel filter over every pixel using either of the sobel X/Y/both 3x3 square filters for a batch of images

// *param[in] srcPtr Input image batch
// *param[in] srcSize Array containing an RppiSize for each image in the batch
// *param[in] maxSrcSize A single RppiSize which is the maxWidth and maxHeight for all images in the batch
// *param[out] dstPtr Output image batch
// *param[in] sobelType Array containing an Rpp32u sobel filter type for each image in the batch (sobelType[n] = 0/1/2)
// *param[in] nbatchSize Batch size or the number of images in the batch
// *param[in] rppHandle OpenCL-handle/HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : No error, Succesful completion
// *retval RPP_ERROR : Error

RppStatus rppi_sobel_filter_u8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *sobelType, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_sobel_filter_u8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *sobelType, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_sobel_filter_u8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *sobelType, Rpp32u nbatchSize, rppHandle_t rppHandle);
#ifdef GPU_SUPPORT
RppStatus rppi_sobel_filter_u8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *sobelType, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_sobel_filter_u8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *sobelType, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_sobel_filter_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *sobelType, Rpp32u nbatchSize, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/******************** median_filter ********************/

// Applies the median filter over every pixel using a [kernelSize X kernelSize] square mask for a batch of images

// *param[in] srcPtr Input image batch
// *param[in] srcSize Array containing an RppiSize for each image in the batch
// *param[in] maxSrcSize A single RppiSize which is the maxWidth and maxHeight for all images in the batch
// *param[out] dstPtr Output image batch
// *param[in] kernelSize Array containing an Rpp32u kernel size for each image in the batch (kernelSize[n] = 3/5/7 for optimal use)
// *param[in] nbatchSize Batch size or the number of images in the batch
// *param[in] rppHandle OpenCL-handle/HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : No error, Succesful completion
// *retval RPP_ERROR : Error

RppStatus rppi_median_filter_u8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *kernelSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_median_filter_u8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *kernelSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_median_filter_u8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *kernelSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
#ifdef GPU_SUPPORT
RppStatus rppi_median_filter_u8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *kernelSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_median_filter_u8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *kernelSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_median_filter_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *kernelSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/******************** custom_convolution ********************/

// Applies the custom convolution filter over every pixel using a filter kernel provided by the user for a batch of images

// *param[in] srcPtr Input image batch
// *param[in] srcSize Array containing an RppiSize for each image in the batch
// *param[in] maxSrcSize A single RppiSize which is the maxWidth and maxHeight for all images in the batch
// *param[out] dstPtr Output image batch
// *param[in] kernel Array containing an [mxn] convolution kernel for each image in the batch
// *param[in] kernelSize Array containing an RppiSize kernel width/height pair for each image in the batch (widths/heights = 3/5/7 for optimal use)
// *param[in] nbatchSize Batch size or the number of images in the batch
// *param[in] rppHandle OpenCL-handle/HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : No error, Succesful completion
// *retval RPP_ERROR : Error

RppStatus rppi_custom_convolution_u8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppPtr_t kernel, RppiSize *kernelSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_custom_convolution_u8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppPtr_t kernel, RppiSize *kernelSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_custom_convolution_u8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppPtr_t kernel, RppiSize *kernelSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
#ifdef GPU_SUPPORT
RppStatus rppi_custom_convolution_u8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppPtr_t kernel, RppiSize *kernelSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_custom_convolution_u8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppPtr_t kernel, RppiSize *kernelSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_custom_convolution_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppPtr_t kernel, RppiSize *kernelSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/******************** non_max_suppression ********************/

// Applies the non-max suppression filter over every pixel using a [kernelSize X kernelSize] square mask for a batch of images

// *param[in] srcPtr Input image batch
// *param[in] srcSize Array containing an RppiSize for each image in the batch
// *param[in] maxSrcSize A single RppiSize which is the maxWidth and maxHeight for all images in the batch
// *param[out] dstPtr Output image batch
// *param[in] kernelSize Array containing an Rpp32u kernel size for each image in the batch (kernelSize[n] = 3/5/7 for optimal use)
// *param[in] nbatchSize Batch size or the number of images in the batch
// *param[in] rppHandle OpenCL-handle/HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : No error, Succesful completion
// *retval RPP_ERROR : Error

RppStatus rppi_non_max_suppression_u8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *kernelSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_non_max_suppression_u8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *kernelSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_non_max_suppression_u8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *kernelSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
#ifdef GPU_SUPPORT
RppStatus rppi_non_max_suppression_u8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *kernelSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_non_max_suppression_u8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *kernelSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_non_max_suppression_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *kernelSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/******************** gaussian_filter ********************/

// Applies the gaussian filter over every pixel using a [kernelSize X kernelSize] square mask for a batch of images

// *param[in] srcPtr Input image batch
// *param[in] srcSize Array containing an RppiSize for each image in the batch
// *param[in] maxSrcSize A single RppiSize which is the maxWidth and maxHeight for all images in the batch
// *param[out] dstPtr Output image batch
// *param[in] stdDev Array containing an Rpp32f standard deviation to populate the kernel for each image in the batch
// *param[in] kernelSize Array containing an Rpp32u kernel size for each image in the batch (kernelSize[n] = 3/5/7 for optimal use)
// *param[in] nbatchSize Batch size or the number of images in the batch
// *param[in] rppHandle OpenCL-handle/HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : No error, Succesful completion
// *retval RPP_ERROR : Error

RppStatus rppi_gaussian_filter_u8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *stdDev, Rpp32u *kernelSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_gaussian_filter_u8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *stdDev, Rpp32u *kernelSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_gaussian_filter_u8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *stdDev, Rpp32u *kernelSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
#ifdef GPU_SUPPORT
RppStatus rppi_gaussian_filter_u8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *stdDev, Rpp32u *kernelSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_gaussian_filter_u8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *stdDev, Rpp32u *kernelSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_gaussian_filter_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *stdDev, Rpp32u *kernelSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/******************** nonlinear_filter ********************/

// Applies the nonlinear filter over every pixel using a [kernelSize X kernelSize] square mask for a batch of images

// *param[in] srcPtr Input image batch
// *param[in] srcSize Array containing an RppiSize for each image in the batch
// *param[in] maxSrcSize A single RppiSize which is the maxWidth and maxHeight for all images in the batch
// *param[out] dstPtr Output image batch
// *param[in] kernelSize Array containing an Rpp32u kernel size for each image in the batch (kernelSize[n] = 3/5/7 for optimal use)
// *param[in] nbatchSize Batch size or the number of images in the batch
// *param[in] rppHandle OpenCL-handle/HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : No error, Succesful completion
// *retval RPP_ERROR : Error

RppStatus rppi_nonlinear_filter_u8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *kernelSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_nonlinear_filter_u8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *kernelSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_nonlinear_filter_u8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *kernelSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
#ifdef GPU_SUPPORT
RppStatus rppi_nonlinear_filter_u8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *kernelSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_nonlinear_filter_u8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *kernelSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_nonlinear_filter_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *kernelSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

#ifdef GPU_SUPPORT
RppStatus nppi_prewitt_filter_u8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtrx, RppPtr_t dstPtry, RppPtr_t pDstMag, RppPtr_t pDstAngle, RppiPoint maskLoc, RppiNorm eNorm, Rpp32u nbatchSize, RppiBorderType rBorderType, rppHandle_t rppHandle);
NppStatus nppiGradientVectorPrewittBorder_8u16s_C1R(const Npp8u *pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s *pDstX, int nDstXStep, Npp16s *pDstY, int nDstYStep, Npp16s *pDstMag, int nDstMagStep, Npp32f *pDstAngle, int nDstAngleStep, NppiSize oSizeROI, NppiMaskSize eMaskSize, NppiNorm eNorm, NppiBorderType eBorderType);
NppStatus nppiLabelMarkersUFGetBufferSize_32u_C1R(NppiSize oSizeROI, int *hpBufferSize);
NppStatus nppiCompressMarkerLabelsGetBufferSize_32u_C1R(int nStartingNumber, int *hpBufferSize);
NppStatus nppiLabelMarkersUF_8u32u_C1R_Ctx(Npp8u *pSrc, int nSrcStep, Npp32u *pDst, int nDstStep, NppiSize oSizeROI, NppiNorm eNorm, Npp8u *pBuffer, NppStreamContext nppStreamCtx);
RppStatus rppi_labelmarkers_u8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiNorm eNorm, Rpp32u nbatchSize, rppHandle_t rppHandle);
NppStatus nppiLabelMarkersUFBatch_8u32u_C1R_Advanced_Ctx(const NppiImageDescriptor *pSrcBatchList, NppiImageDescriptor *pDstBatchList, int nBatchSize, NppiSize oMaxSizeROI, NppiNorm eNorm, NppStreamContext nppStreamCtx);
NppStatus nppiCompressMarkerLabelsUFBatch_32u_C1IR_Advanced_Ctx(NppiImageDescriptor *pSrcDstBatchList, NppiBufferDescriptor *pBufferList, unsigned int *pNewMaxLabelIDList,
									int nBatchSize, NppiSize oMaxSizeROI,int nLargestPerImageBufferSize, NppStreamContext nppStreamCtx);
RppStatus rppi_compressmarkers_u8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppBufferDescriptor *pBufferBatch, Rpp32u *pNewMaxLabelID, Rpp32s nPerImageBufferSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
NppStatus nppiCompressMarkerLabelsUF_32u_C1IR(Npp32u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nStartingNumber, int *pNewNumber, Npp8u *pBuffer);
RppStatus rppi_compressmarkers_u8_pln1_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, Rpp32s *pBufferBatch, Rpp32s nStartingNumber, Rpp32u nbatchSize, rppHandle_t rppHandle);
NppStatus nppiFilterBoxBorder_8u_C1R(const Npp8u *pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u *pDst, Npp32s nDstStep, 
									NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType);
RppStatus nppi_box_filter_u8_pln1_batchPD_gpu(RppPtr_t srcPtr,RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u *kernelSize, RppiPoint maskAnchor,RppiPoint maskLoc, Rpp32u nbatchSize, RppiBorderType rBorderType, rppHandle_t handle);
NppStatus nppiFilterSobelHoriz_8u_C3R(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep, NppiSize oSizeROI);
NppStatus nppiFilterSobelHoriz_8u_C1R_Ctx(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus nppiFilterSobelHoriz_8u_C1R(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep, NppiSize oSizeROI);
NppStatus nppiFilterSobelHoriz_8u_C3R_Ctx(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus nppiFilterSobelVert_8u_C3R(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep, NppiSize oSizeROI);
NppStatus nppiFilterSobelVert_8u_C1R_Ctx(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus nppiFilterSobelVert_8u_C1R(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep, NppiSize oSizeROI);
NppStatus nppiFilterSobelVert_8u_C3R_Ctx(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus nppiFilterMedian_8u_C3R(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor, Npp8u *pBuffer);
NppStatus nppiFilterMedian_8u_C1R_Ctx(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor, Npp8u *pBuffer, NppStreamContext nppStreamCtx);
NppStatus nppiFilterMedian_8u_C1R(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor, Npp8u *pBuffer);
NppStatus nppiFilterMedian_8u_C3R_Ctx(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor, Npp8u *pBuffer, NppStreamContext nppStreamCtx);
NppStatus nppiFilterGauss_8u_C3R(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiMaskSize eMaskSize);
NppStatus nppiFilterGauss_8u_C1R_Ctx(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiMaskSize eMaskSize, NppStreamContext nppStreamCtx);
NppStatus nppiFilterGauss_8u_C1R(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiMaskSize eMaskSize);
NppStatus nppiFilterGauss_8u_C3R_Ctx(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiMaskSize eMaskSize, NppStreamContext nppStreamCtx);
#endif // GPU_SUPPORT

/*! @}
 */

#ifdef __cplusplus
}
#endif
#endif

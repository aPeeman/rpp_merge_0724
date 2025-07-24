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
#include "rppt_tensor_geometric_augmentations.h"
#include "cpu/host_tensor_geometric_augmentations.hpp"

#ifdef HIP_COMPILE
#include <hip/hip_fp16.h>
#include "hip/hip_tensor_geometric_augmentations.hpp"
#endif // HIP_COMPILE

#if __APPLE__
#define sincosf __sincosf
#endif

/******************** crop ********************/

RppStatus rppt_crop_host(RppPtr_t srcPtr,
                         RpptDescPtr srcDescPtr,
                         RppPtr_t dstPtr,
                         RpptDescPtr dstDescPtr,
                         RpptROIPtr roiTensorPtrSrc,
                         RpptRoiType roiType,
                         rppHandle_t rppHandle)
{
    RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        crop_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                               srcDescPtr,
                               static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                               dstDescPtr,
                               roiTensorPtrSrc,
                               roiType,
                               layoutParams,
                               rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        crop_f16_f16_host_tensor((Rpp16f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                 srcDescPtr,
                                 (Rpp16f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                 dstDescPtr,
                                 roiTensorPtrSrc,
                                 roiType,
                                 layoutParams,
                                 rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        crop_f32_f32_host_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                 srcDescPtr,
                                 (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                 dstDescPtr,
                                 roiTensorPtrSrc,
                                 roiType,
                                 layoutParams,
                                 rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        crop_i8_i8_host_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                               srcDescPtr,
                               static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                               dstDescPtr,
                               roiTensorPtrSrc,
                               roiType,
                               layoutParams,
                               rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
}

/******************** crop_and_patch ********************/

RppStatus rppt_crop_and_patch_host(RppPtr_t srcPtr1,
                                   RppPtr_t srcPtr2,
                                   RpptDescPtr srcDescPtr,
                                   RppPtr_t dstPtr,
                                   RpptDescPtr dstDescPtr,
                                   RpptROIPtr roiTensorPtrDst,
                                   RpptROIPtr cropRoi,
                                   RpptROIPtr patchRoi,
                                   RpptRoiType roiType,
                                   rppHandle_t rppHandle)
{
    RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        crop_and_patch_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr1) + srcDescPtr->offsetInBytes,
                                         static_cast<Rpp8u*>(srcPtr2) + srcDescPtr->offsetInBytes,
                                         srcDescPtr,
                                         static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                         dstDescPtr,
                                         roiTensorPtrDst,
                                         cropRoi,
                                         patchRoi,
                                         roiType,
                                         layoutParams,
                                         rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        crop_and_patch_f16_f16_host_tensor((Rpp16f*) (static_cast<Rpp8u*>(srcPtr1) + srcDescPtr->offsetInBytes),
                                           (Rpp16f*) (static_cast<Rpp8u*>(srcPtr2) + srcDescPtr->offsetInBytes),
                                           srcDescPtr,
                                           (Rpp16f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                           dstDescPtr,
                                           roiTensorPtrDst,
                                           cropRoi,
                                           patchRoi,
                                           roiType,
                                           layoutParams,
                                           rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        crop_and_patch_f32_f32_host_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr1) + srcDescPtr->offsetInBytes),
                                           (Rpp32f*) (static_cast<Rpp8u*>(srcPtr2) + srcDescPtr->offsetInBytes),
                                           srcDescPtr,
                                           (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                           dstDescPtr,
                                           roiTensorPtrDst,
                                           cropRoi,
                                           patchRoi,
                                           roiType,
                                           layoutParams,
                                           rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        crop_and_patch_i8_i8_host_tensor(static_cast<Rpp8s*>(srcPtr1) + srcDescPtr->offsetInBytes,
                                         static_cast<Rpp8s*>(srcPtr2) + srcDescPtr->offsetInBytes,
                                         srcDescPtr,
                                         static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                         dstDescPtr,
                                         roiTensorPtrDst,
                                         cropRoi,
                                         patchRoi,
                                         roiType,
                                         layoutParams,
                                         rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
}

/******************** crop mirror normalize ********************/

RppStatus rppt_crop_mirror_normalize_host(RppPtr_t srcPtr,
                                          RpptDescPtr srcDescPtr,
                                          RppPtr_t dstPtr,
                                          RpptDescPtr dstDescPtr,
                                          Rpp32f *offsetTensor,
                                          Rpp32f *multiplierTensor,
                                          Rpp32u *mirrorTensor,
                                          RpptROIPtr roiTensorPtrSrc,
                                          RpptRoiType roiType,
                                          rppHandle_t rppHandle)
{
    RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        crop_mirror_normalize_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                                srcDescPtr,
                                                static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                                dstDescPtr,
                                                offsetTensor,
                                                multiplierTensor,
                                                mirrorTensor,
                                                roiTensorPtrSrc,
                                                roiType,
                                                layoutParams,
                                                rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        crop_mirror_normalize_f16_f16_host_tensor((Rpp16f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                                  srcDescPtr,
                                                  (Rpp16f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                                  dstDescPtr,
                                                  offsetTensor,
                                                  multiplierTensor,
                                                  mirrorTensor,
                                                  roiTensorPtrSrc,
                                                  roiType,
                                                  layoutParams,
                                                  rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        crop_mirror_normalize_f32_f32_host_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                                  srcDescPtr,
                                                  (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                                  dstDescPtr,
                                                  offsetTensor,
                                                  multiplierTensor,
                                                  mirrorTensor,
                                                  roiTensorPtrSrc,
                                                  roiType,
                                                  layoutParams,
                                                  rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        crop_mirror_normalize_i8_i8_host_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                                srcDescPtr,
                                                static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                                dstDescPtr,
                                                offsetTensor,
                                                multiplierTensor,
                                                mirrorTensor,
                                                roiTensorPtrSrc,
                                                roiType,
                                                layoutParams,
                                                rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        crop_mirror_normalize_u8_f32_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                                 srcDescPtr,
                                                 (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                                 dstDescPtr,
                                                 offsetTensor,
                                                 multiplierTensor,
                                                 mirrorTensor,
                                                 roiTensorPtrSrc,
                                                 roiType,
                                                 layoutParams,
                                                 rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        crop_mirror_normalize_u8_f16_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                                 srcDescPtr,
                                                 (Rpp16f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                                 dstDescPtr,
                                                 offsetTensor,
                                                 multiplierTensor,
                                                 mirrorTensor,
                                                 roiTensorPtrSrc,
                                                 roiType,
                                                 layoutParams,
                                                 rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
}

/******************** warp_affine ********************/

RppStatus rppt_warp_affine_host(RppPtr_t srcPtr,
                                RpptDescPtr srcDescPtr,
                                RppPtr_t dstPtr,
                                RpptDescPtr dstDescPtr,
                                Rpp32f *affineTensor,
                                RpptInterpolationType interpolationType,
                                RpptROIPtr roiTensorPtrSrc,
                                RpptRoiType roiType,
                                rppHandle_t rppHandle)
{
    if ((interpolationType != RpptInterpolationType::BILINEAR) && (interpolationType != RpptInterpolationType::NEAREST_NEIGHBOR))
        return RPP_ERROR_NOT_IMPLEMENTED;

    RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);

    if(interpolationType == RpptInterpolationType::NEAREST_NEIGHBOR)
    {
        if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
        {
            warp_affine_nn_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                             srcDescPtr,
                                             static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                             dstDescPtr,
                                             affineTensor,
                                             roiTensorPtrSrc,
                                             roiType,
                                             layoutParams,
                                             rpp::deref(rppHandle));
        }
        else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
        {
            warp_affine_nn_f16_f16_host_tensor((Rpp16f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                               srcDescPtr,
                                               (Rpp16f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                               dstDescPtr,
                                               affineTensor,
                                               roiTensorPtrSrc,
                                               roiType,
                                               layoutParams,
                                               rpp::deref(rppHandle));
        }
        else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
        {
            warp_affine_nn_f32_f32_host_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                               srcDescPtr,
                                               (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                               dstDescPtr,
                                               affineTensor,
                                               roiTensorPtrSrc,
                                               roiType,
                                               layoutParams,
                                               rpp::deref(rppHandle));
        }
        else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
        {
            warp_affine_nn_i8_i8_host_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                             srcDescPtr,
                                             static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                             dstDescPtr,
                                             affineTensor,
                                             roiTensorPtrSrc,
                                             roiType,
                                             layoutParams,
                                             rpp::deref(rppHandle));
        }
    }
    else if(interpolationType == RpptInterpolationType::BILINEAR)
    {
        if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
        {
            warp_affine_bilinear_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                                   srcDescPtr,
                                                   static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                                   dstDescPtr,
                                                   affineTensor,
                                                   roiTensorPtrSrc,
                                                   roiType,
                                                   layoutParams,
                                                   rpp::deref(rppHandle));
        }
        else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
        {
            warp_affine_bilinear_f16_f16_host_tensor((Rpp16f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                                     srcDescPtr,
                                                     (Rpp16f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                                     dstDescPtr,
                                                     affineTensor,
                                                     roiTensorPtrSrc,
                                                     roiType,
                                                     layoutParams,
                                                     rpp::deref(rppHandle));
        }
        else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
        {
            warp_affine_bilinear_f32_f32_host_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                                     srcDescPtr,
                                                     (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                                     dstDescPtr,
                                                     affineTensor,
                                                     roiTensorPtrSrc,
                                                     roiType,
                                                     layoutParams,
                                                     rpp::deref(rppHandle));
        }
        else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
        {
            warp_affine_bilinear_i8_i8_host_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                                   srcDescPtr,
                                                   static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                                   dstDescPtr,
                                                   affineTensor,
                                                   roiTensorPtrSrc,
                                                   roiType,
                                                   layoutParams,
                                                   rpp::deref(rppHandle));
        }
    }

    return RPP_SUCCESS;
}

/******************** flip ********************/

RppStatus rppt_flip_host(RppPtr_t srcPtr,
                         RpptDescPtr srcDescPtr,
                         RppPtr_t dstPtr,
                         RpptDescPtr dstDescPtr,
                         Rpp32u *horizontalTensor,
                         Rpp32u *verticalTensor,
                         RpptROIPtr roiTensorPtrSrc,
                         RpptRoiType roiType,
                         rppHandle_t rppHandle)
{
    RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        flip_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                               srcDescPtr,
                               static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                               dstDescPtr,
                               horizontalTensor,
                               verticalTensor,
                               roiTensorPtrSrc,
                               roiType,
                               layoutParams,
                               rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        flip_f16_f16_host_tensor((Rpp16f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                 srcDescPtr,
                                 (Rpp16f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                 dstDescPtr,
                                 horizontalTensor,
                                 verticalTensor,
                                 roiTensorPtrSrc,
                                 roiType,
                                 layoutParams,
                                 rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        flip_f32_f32_host_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                 srcDescPtr,
                                 (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                 dstDescPtr,
                                 horizontalTensor,
                                 verticalTensor,
                                 roiTensorPtrSrc,
                                 roiType,
                                 layoutParams,
                                 rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        flip_i8_i8_host_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                               srcDescPtr,
                               static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                               dstDescPtr,
                               horizontalTensor,
                               verticalTensor,
                               roiTensorPtrSrc,
                               roiType,
                               layoutParams,
                               rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
}

/******************** resize ********************/

RppStatus rppt_resize_host(RppPtr_t srcPtr,
                           RpptDescPtr srcDescPtr,
                           RppPtr_t dstPtr,
                           RpptDescPtr dstDescPtr,
                           RpptImagePatchPtr dstImgSizes,
                           RpptInterpolationType interpolationType,
                           RpptROIPtr roiTensorPtrSrc,
                           RpptRoiType roiType,
                           rppHandle_t rppHandle)
{
    RppLayoutParams srcLayoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);

    if(interpolationType == RpptInterpolationType::NEAREST_NEIGHBOR)
    {
        if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
        {
            resize_nn_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                        srcDescPtr,
                                        static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                        dstDescPtr,
                                        dstImgSizes,
                                        roiTensorPtrSrc,
                                        roiType,
                                        srcLayoutParams,
                                        rpp::deref(rppHandle));
        }
        else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
        {
            resize_nn_f16_f16_host_tensor((Rpp16f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                          srcDescPtr,
                                          (Rpp16f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                          dstDescPtr,
                                          dstImgSizes,
                                          roiTensorPtrSrc,
                                          roiType,
                                          srcLayoutParams,
                                          rpp::deref(rppHandle));
        }
        else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
        {
            resize_nn_f32_f32_host_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                          srcDescPtr,
                                          (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                          dstDescPtr,
                                          dstImgSizes,
                                          roiTensorPtrSrc,
                                          roiType,
                                          srcLayoutParams,
                                          rpp::deref(rppHandle));
        }
        else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
        {
            resize_nn_i8_i8_host_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                        srcDescPtr,
                                        static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                        dstDescPtr,
                                        dstImgSizes,
                                        roiTensorPtrSrc,
                                        roiType,
                                        srcLayoutParams,
                                        rpp::deref(rppHandle));
        }
    }
    else if(interpolationType == RpptInterpolationType::BILINEAR)
    {
        if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
        {
            resize_bilinear_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                              srcDescPtr,
                                              static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                              dstDescPtr,
                                              dstImgSizes,
                                              roiTensorPtrSrc,
                                              roiType,
                                              srcLayoutParams,
                                              rpp::deref(rppHandle));
        }
        else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
        {
            resize_bilinear_f16_f16_host_tensor((Rpp16f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                                srcDescPtr,
                                                (Rpp16f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                                dstDescPtr,
                                                dstImgSizes,
                                                roiTensorPtrSrc,
                                                roiType,
                                                srcLayoutParams,
                                                rpp::deref(rppHandle));
        }
        else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
        {
            resize_bilinear_f32_f32_host_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                                srcDescPtr,
                                                (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                                dstDescPtr,
                                                dstImgSizes,
                                                roiTensorPtrSrc,
                                                roiType,
                                                srcLayoutParams,
                                                rpp::deref(rppHandle));
        }
        else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
        {
            resize_bilinear_i8_i8_host_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                               srcDescPtr,
                                               static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                               dstDescPtr,
                                               dstImgSizes,
                                               roiTensorPtrSrc,
                                               roiType,
                                               srcLayoutParams,
                                               rpp::deref(rppHandle));
        }
    }
    else
    {
        RpptDesc tempDesc;
        tempDesc = *srcDescPtr;
        RpptDescPtr tempDescPtr = &tempDesc;
        tempDescPtr->h = dstDescPtr->h;
        tempDescPtr->strides.nStride = srcDescPtr->w * dstDescPtr->h * srcDescPtr->c;

        // The channel stride changes with the change in the height for PLN images
        if(srcDescPtr->layout == RpptLayout::NCHW)
            tempDescPtr->strides.cStride = srcDescPtr->w * dstDescPtr->h;

        if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
        {
            resize_separable_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                         srcDescPtr,
                                         static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                         dstDescPtr,
                                         rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.scratchBufferHost,
                                         tempDescPtr,
                                         dstImgSizes,
                                         roiTensorPtrSrc,
                                         roiType,
                                         srcLayoutParams,
                                         interpolationType,
                                         rpp::deref(rppHandle));
        }
        else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
        {
            resize_separable_host_tensor(static_cast<Rpp32f*>(srcPtr) + srcDescPtr->offsetInBytes,
                                         srcDescPtr,
                                         static_cast<Rpp32f*>(dstPtr) + dstDescPtr->offsetInBytes,
                                         dstDescPtr,
                                         rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.scratchBufferHost,
                                         tempDescPtr,
                                         dstImgSizes,
                                         roiTensorPtrSrc,
                                         roiType,
                                         srcLayoutParams,
                                         interpolationType,
                                         rpp::deref(rppHandle));
        }
        else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
        {
            resize_separable_host_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                         srcDescPtr,
                                         static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                         dstDescPtr,
                                         rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.scratchBufferHost,
                                         tempDescPtr,
                                         dstImgSizes,
                                         roiTensorPtrSrc,
                                         roiType,
                                         srcLayoutParams,
                                         interpolationType,
                                         rpp::deref(rppHandle));
        }
        else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
        {
            resize_separable_host_tensor(static_cast<Rpp16f*>(srcPtr) + srcDescPtr->offsetInBytes,
                                         srcDescPtr,
                                         static_cast<Rpp16f*>(dstPtr) + dstDescPtr->offsetInBytes,
                                         dstDescPtr,
                                         rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.scratchBufferHost,
                                         tempDescPtr,
                                         dstImgSizes,
                                         roiTensorPtrSrc,
                                         roiType,
                                         srcLayoutParams,
                                         interpolationType,
                                         rpp::deref(rppHandle));
        }
    }

    return RPP_SUCCESS;
}

/******************** resize_mirror_normalize ********************/

RppStatus rppt_resize_mirror_normalize_host(RppPtr_t srcPtr,
                                            RpptDescPtr srcDescPtr,
                                            RppPtr_t dstPtr,
                                            RpptDescPtr dstDescPtr,
                                            RpptImagePatchPtr dstImgSizes,
                                            RpptInterpolationType interpolationType,
                                            Rpp32f *meanTensor,
                                            Rpp32f *stdDevTensor,
                                            Rpp32u *mirrorTensor,
                                            RpptROIPtr roiTensorPtrSrc,
                                            RpptRoiType roiType,
                                            rppHandle_t rppHandle)
{
    RppLayoutParams srcLayoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);

    if (interpolationType != RpptInterpolationType::BILINEAR)
        return RPP_ERROR_NOT_IMPLEMENTED;

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        resize_mirror_normalize_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                                  srcDescPtr,
                                                  static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                                  dstDescPtr,
                                                  dstImgSizes,
                                                  meanTensor,
                                                  stdDevTensor,
                                                  mirrorTensor,
                                                  roiTensorPtrSrc,
                                                  roiType,
                                                  srcLayoutParams,
                                                  rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        resize_mirror_normalize_f16_f16_host_tensor((Rpp16f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                                    srcDescPtr,
                                                    (Rpp16f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                                    dstDescPtr,
                                                    dstImgSizes,
                                                    meanTensor,
                                                    stdDevTensor,
                                                    mirrorTensor,
                                                    roiTensorPtrSrc,
                                                    roiType,
                                                    srcLayoutParams,
                                                    rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        resize_mirror_normalize_f32_f32_host_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                                    srcDescPtr,
                                                    (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                                    dstDescPtr,
                                                    dstImgSizes,
                                                    meanTensor,
                                                    stdDevTensor,
                                                    mirrorTensor,
                                                    roiTensorPtrSrc,
                                                    roiType,
                                                    srcLayoutParams,
                                                    rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        resize_mirror_normalize_i8_i8_host_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                                  srcDescPtr,
                                                  static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                                  dstDescPtr,
                                                  dstImgSizes,
                                                  meanTensor,
                                                  stdDevTensor,
                                                  mirrorTensor,
                                                  roiTensorPtrSrc,
                                                  roiType,
                                                  srcLayoutParams,
                                                  rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        resize_mirror_normalize_u8_f32_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                                   srcDescPtr,
                                                   (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                                   dstDescPtr,
                                                   dstImgSizes,
                                                   meanTensor,
                                                   stdDevTensor,
                                                   mirrorTensor,
                                                   roiTensorPtrSrc,
                                                   roiType,
                                                   srcLayoutParams,
                                                   rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        resize_mirror_normalize_u8_f16_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                                   srcDescPtr,
                                                   (Rpp16f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                                   dstDescPtr,
                                                   dstImgSizes,
                                                   meanTensor,
                                                   stdDevTensor,
                                                   mirrorTensor,
                                                   roiTensorPtrSrc,
                                                   roiType,
                                                   srcLayoutParams,
                                                   rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
}

RppStatus rppt_resize_crop_mirror_host(RppPtr_t srcPtr,
                                       RpptDescPtr srcDescPtr,
                                       RppPtr_t dstPtr,
                                       RpptDescPtr dstDescPtr,
                                       RpptImagePatchPtr dstImgSizes,
                                       RpptInterpolationType interpolationType,
                                       Rpp32u *mirrorTensor,
                                       RpptROIPtr roiTensorPtrSrc,
                                       RpptRoiType roiType,
                                       rppHandle_t rppHandle)
{
    RppLayoutParams srcLayoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);

    if (interpolationType != RpptInterpolationType::BILINEAR)
        return RPP_ERROR_NOT_IMPLEMENTED;

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        resize_crop_mirror_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                             srcDescPtr,
                                             static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                             dstDescPtr,
                                             dstImgSizes,
                                             mirrorTensor,
                                             roiTensorPtrSrc,
                                             roiType,
                                             srcLayoutParams,
                                             rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        resize_crop_mirror_f16_f16_host_tensor((Rpp16f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                               srcDescPtr,
                                               (Rpp16f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                               dstDescPtr,
                                               dstImgSizes,
                                               mirrorTensor,
                                               roiTensorPtrSrc,
                                               roiType,
                                               srcLayoutParams,
                                               rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        resize_crop_mirror_f32_f32_host_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                               srcDescPtr,
                                               (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                               dstDescPtr,
                                               dstImgSizes,
                                               mirrorTensor,
                                               roiTensorPtrSrc,
                                               roiType,
                                               srcLayoutParams,
                                               rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        resize_crop_mirror_i8_i8_host_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                             srcDescPtr,
                                             static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                             dstDescPtr,
                                             dstImgSizes,
                                             mirrorTensor,
                                             roiTensorPtrSrc,
                                             roiType,
                                             srcLayoutParams,
                                             rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
}

/******************** rotate ********************/

RppStatus rppt_rotate_host(RppPtr_t srcPtr,
                           RpptDescPtr srcDescPtr,
                           RppPtr_t dstPtr,
                           RpptDescPtr dstDescPtr,
                           Rpp32f *angle,
                           RpptInterpolationType interpolationType,
                           RpptROIPtr roiTensorPtrSrc,
                           RpptRoiType roiType,
                           rppHandle_t rppHandle)
{
    if ((interpolationType != RpptInterpolationType::BILINEAR) && (interpolationType != RpptInterpolationType::NEAREST_NEIGHBOR))
        return RPP_ERROR_NOT_IMPLEMENTED;

    RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);

    // Compute affine transformation matrix from rotate angle
    Rpp32f *affineTensor = rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.scratchBufferHost;
    for(int idx = 0; idx < srcDescPtr->n; idx++)
    {
        Rpp32f angleInRad = angle[idx] * PI_OVER_180;
        Rpp32f alpha, beta;
        sincosf(angleInRad, &beta, &alpha);
        ((Rpp32f6 *)affineTensor)[idx] = {alpha, -beta, 0, beta, alpha, 0};
    }

    if(interpolationType == RpptInterpolationType::NEAREST_NEIGHBOR)
    {
        if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
        {
            warp_affine_nn_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                             srcDescPtr,
                                             static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                             dstDescPtr,
                                             affineTensor,
                                             roiTensorPtrSrc,
                                             roiType,
                                             layoutParams,
                                             rpp::deref(rppHandle));
        }
        else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
        {
            warp_affine_nn_f16_f16_host_tensor((Rpp16f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                               srcDescPtr,
                                               (Rpp16f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                               dstDescPtr,
                                               affineTensor,
                                               roiTensorPtrSrc,
                                               roiType,
                                               layoutParams,
                                               rpp::deref(rppHandle));
        }
        else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
        {
            warp_affine_nn_f32_f32_host_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                               srcDescPtr,
                                               (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                               dstDescPtr,
                                               affineTensor,
                                               roiTensorPtrSrc,
                                               roiType,
                                               layoutParams,
                                               rpp::deref(rppHandle));
        }
        else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
        {
            warp_affine_nn_i8_i8_host_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                             srcDescPtr,
                                             static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                             dstDescPtr,
                                             affineTensor,
                                             roiTensorPtrSrc,
                                             roiType,
                                             layoutParams,
                                             rpp::deref(rppHandle));
        }
    }
    else if(interpolationType == RpptInterpolationType::BILINEAR)
    {
        if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
        {
            warp_affine_bilinear_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                                   srcDescPtr,
                                                   static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                                   dstDescPtr,
                                                   affineTensor,
                                                   roiTensorPtrSrc,
                                                   roiType,
                                                   layoutParams,
                                                   rpp::deref(rppHandle));
        }
        else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
        {
            warp_affine_bilinear_f16_f16_host_tensor((Rpp16f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                                     srcDescPtr,
                                                     (Rpp16f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                                     dstDescPtr,
                                                     affineTensor,
                                                     roiTensorPtrSrc,
                                                     roiType,
                                                     layoutParams,
                                                     rpp::deref(rppHandle));
        }
        else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
        {
            warp_affine_bilinear_f32_f32_host_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                                     srcDescPtr,
                                                     (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                                     dstDescPtr,
                                                     affineTensor,
                                                     roiTensorPtrSrc,
                                                     roiType,
                                                     layoutParams,
                                                     rpp::deref(rppHandle));
        }
        else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
        {
            warp_affine_bilinear_i8_i8_host_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                                   srcDescPtr,
                                                   static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                                   dstDescPtr,
                                                   affineTensor,
                                                   roiTensorPtrSrc,
                                                   roiType,
                                                   layoutParams,
                                                   rpp::deref(rppHandle));
        }
    }

    return RPP_SUCCESS;
}

/******************** phase ********************/

RppStatus rppt_phase_host(RppPtr_t srcPtr1,
                          RppPtr_t srcPtr2,
                          RpptDescPtr srcDescPtr,
                          RppPtr_t dstPtr,
                          RpptDescPtr dstDescPtr,
                          RpptROIPtr roiTensorPtrSrc,
                          RpptRoiType roiType,
                          rppHandle_t rppHandle)
{
    RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        phase_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr1) + srcDescPtr->offsetInBytes,
                                static_cast<Rpp8u*>(srcPtr2) + srcDescPtr->offsetInBytes,
                                srcDescPtr,
                                static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                dstDescPtr,
                                roiTensorPtrSrc,
                                roiType,
                                layoutParams,
                                rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        phase_f16_f16_host_tensor(reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(srcPtr1) + srcDescPtr->offsetInBytes),
                                  reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(srcPtr2) + srcDescPtr->offsetInBytes),
                                  srcDescPtr,
                                  reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                  dstDescPtr,
                                  roiTensorPtrSrc,
                                  roiType,
                                  layoutParams,
                                  rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        phase_f32_f32_host_tensor(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr1) + srcDescPtr->offsetInBytes),
                                  reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr2) + srcDescPtr->offsetInBytes),
                                  srcDescPtr,
                                  reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                  dstDescPtr,
                                  roiTensorPtrSrc,
                                  roiType,
                                  layoutParams,
                                  rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        phase_i8_i8_host_tensor(static_cast<Rpp8s*>(srcPtr1) + srcDescPtr->offsetInBytes,
                                static_cast<Rpp8s*>(srcPtr2) + srcDescPtr->offsetInBytes,
                                srcDescPtr,
                                static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                dstDescPtr,
                                roiTensorPtrSrc,
                                roiType,
                                layoutParams,
                                rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
}

/******************** slice ********************/

RppStatus rppt_slice_host(RppPtr_t srcPtr,
                          RpptGenericDescPtr srcGenericDescPtr,
                          RppPtr_t dstPtr,
                          RpptGenericDescPtr dstGenericDescPtr,
                          RpptROI3DPtr roiGenericPtrSrc,
                          RpptRoi3DType roiType,
                          rppHandle_t rppHandle)
{
    RppLayoutParams layoutParams;
    if ((srcGenericDescPtr->layout == RpptLayout::NCDHW) && (dstGenericDescPtr->layout == RpptLayout::NCDHW))
        layoutParams = get_layout_params(srcGenericDescPtr->layout, srcGenericDescPtr->dims[1]);
    else if ((srcGenericDescPtr->layout == RpptLayout::NDHWC) && (dstGenericDescPtr->layout == RpptLayout::NDHWC))
        layoutParams = get_layout_params(srcGenericDescPtr->layout, srcGenericDescPtr->dims[4]);

    if ((srcGenericDescPtr->dataType != RpptDataType::F32) && (srcGenericDescPtr->dataType != RpptDataType::U8)) return RPP_ERROR_INVALID_SRC_DATATYPE;
    if ((dstGenericDescPtr->dataType != RpptDataType::F32) && (dstGenericDescPtr->dataType != RpptDataType::U8)) return RPP_ERROR_INVALID_DST_DATATYPE;
    if ((srcGenericDescPtr->layout != RpptLayout::NCDHW) && (srcGenericDescPtr->layout != RpptLayout::NDHWC)) return RPP_ERROR_INVALID_SRC_LAYOUT;
    if ((dstGenericDescPtr->layout != RpptLayout::NCDHW) && (dstGenericDescPtr->layout != RpptLayout::NDHWC)) return RPP_ERROR_INVALID_DST_LAYOUT;
    if (srcGenericDescPtr->layout != dstGenericDescPtr->layout) return RPP_ERROR_INVALID_ARGUMENTS;

    if ((srcGenericDescPtr->dataType == RpptDataType::F32) && (dstGenericDescPtr->dataType == RpptDataType::F32))
    {
        slice_f32_f32_host_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcGenericDescPtr->offsetInBytes),
                                  srcGenericDescPtr,
                                  (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstGenericDescPtr->offsetInBytes),
                                  dstGenericDescPtr,
                                  roiGenericPtrSrc,
                                  roiType,
                                  layoutParams,
                                  rpp::deref(rppHandle));
    }
    else if ((srcGenericDescPtr->dataType == RpptDataType::U8) && (dstGenericDescPtr->dataType == RpptDataType::U8))
    {
        slice_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcGenericDescPtr->offsetInBytes,
                                srcGenericDescPtr,
                                static_cast<Rpp8u*>(dstPtr) + dstGenericDescPtr->offsetInBytes,
                                dstGenericDescPtr,
                                roiGenericPtrSrc,
                                roiType,
                                layoutParams,
                                rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
}

/******************** flip_voxel ********************/

RppStatus rppt_flip_voxel_host(RppPtr_t srcPtr,
                               RpptGenericDescPtr srcGenericDescPtr,
                               RppPtr_t dstPtr,
                               RpptGenericDescPtr dstGenericDescPtr,
                               Rpp32u *horizontalTensor,
                               Rpp32u *verticalTensor,
                               Rpp32u *depthTensor,
                               RpptROI3DPtr roiGenericPtrSrc,
                               RpptRoi3DType roiType,
                               rppHandle_t rppHandle)
{
    RppLayoutParams layoutParams;
    if ((srcGenericDescPtr->layout == RpptLayout::NCDHW) && (dstGenericDescPtr->layout == RpptLayout::NCDHW))
        layoutParams = get_layout_params(srcGenericDescPtr->layout, srcGenericDescPtr->dims[1]);
    else if ((srcGenericDescPtr->layout == RpptLayout::NDHWC) && (dstGenericDescPtr->layout == RpptLayout::NDHWC))
        layoutParams = get_layout_params(srcGenericDescPtr->layout, srcGenericDescPtr->dims[4]);

    if ((srcGenericDescPtr->dataType != RpptDataType::F32) && (srcGenericDescPtr->dataType != RpptDataType::U8)) return RPP_ERROR_INVALID_SRC_DATATYPE;
    if ((dstGenericDescPtr->dataType != RpptDataType::F32) && (dstGenericDescPtr->dataType != RpptDataType::U8)) return RPP_ERROR_INVALID_DST_DATATYPE;
    if ((srcGenericDescPtr->layout != RpptLayout::NCDHW) && (srcGenericDescPtr->layout != RpptLayout::NDHWC)) return RPP_ERROR_INVALID_SRC_LAYOUT;
    if ((dstGenericDescPtr->layout != RpptLayout::NCDHW) && (dstGenericDescPtr->layout != RpptLayout::NDHWC)) return RPP_ERROR_INVALID_DST_LAYOUT;
    if (srcGenericDescPtr->layout != dstGenericDescPtr->layout) return RPP_ERROR_INVALID_ARGUMENTS;

    if ((srcGenericDescPtr->dataType == RpptDataType::F32) && (dstGenericDescPtr->dataType == RpptDataType::F32))
    {
        flip_voxel_f32_f32_host_tensor(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcGenericDescPtr->offsetInBytes),
                                       srcGenericDescPtr,
                                       reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstGenericDescPtr->offsetInBytes),
                                       dstGenericDescPtr,
                                       horizontalTensor,
                                       verticalTensor,
                                       depthTensor,
                                       roiGenericPtrSrc,
                                       roiType,
                                       layoutParams,
                                       rpp::deref(rppHandle));
    }
    else if ((srcGenericDescPtr->dataType == RpptDataType::U8) && (dstGenericDescPtr->dataType == RpptDataType::U8))
    {
        flip_voxel_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcGenericDescPtr->offsetInBytes,
                                     srcGenericDescPtr,
                                     static_cast<Rpp8u*>(dstPtr) + dstGenericDescPtr->offsetInBytes,
                                     dstGenericDescPtr,
                                     horizontalTensor,
                                     verticalTensor,
                                     depthTensor,
                                     roiGenericPtrSrc,
                                     roiType,
                                     layoutParams,
                                     rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
}

/******************** remap ********************/

RppStatus rppt_remap_host(RppPtr_t srcPtr,
                          RpptDescPtr srcDescPtr,
                          RppPtr_t dstPtr,
                          RpptDescPtr dstDescPtr,
                          Rpp32f *rowRemapTable,
                          Rpp32f *colRemapTable,
                          RpptDescPtr tableDescPtr,
                          RpptInterpolationType interpolationType,
                          RpptROIPtr roiTensorPtrSrc,
                          RpptRoiType roiType,
                          rppHandle_t rppHandle)
{
    RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);

    if (interpolationType != RpptInterpolationType::NEAREST_NEIGHBOR && interpolationType != RpptInterpolationType::BILINEAR)
        return RPP_ERROR_NOT_IMPLEMENTED;

    if(interpolationType == RpptInterpolationType::NEAREST_NEIGHBOR)
    {
        if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
        {
            remap_nn_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                       srcDescPtr,
                                       static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                       dstDescPtr,
                                       rowRemapTable,
                                       colRemapTable,
                                       tableDescPtr,
                                       roiTensorPtrSrc,
                                       roiType,
                                       layoutParams,
                                       rpp::deref(rppHandle));
        }
        else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
        {
            remap_nn_f16_f16_host_tensor(reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                         srcDescPtr,
                                         reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                         dstDescPtr,
                                         rowRemapTable,
                                         colRemapTable,
                                         tableDescPtr,
                                         roiTensorPtrSrc,
                                         roiType,
                                         layoutParams,
                                         rpp::deref(rppHandle));
        }
        else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
        {
            remap_nn_f32_f32_host_tensor(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                         srcDescPtr,
                                         reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                         dstDescPtr,
                                         rowRemapTable,
                                         colRemapTable,
                                         tableDescPtr,
                                         roiTensorPtrSrc,
                                         roiType,
                                         layoutParams,
                                         rpp::deref(rppHandle));
        }
        else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
        {
            remap_nn_i8_i8_host_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                       srcDescPtr,
                                       static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                       dstDescPtr,
                                       rowRemapTable,
                                       colRemapTable,
                                       tableDescPtr,
                                       roiTensorPtrSrc,
                                       roiType,
                                       layoutParams,
                                       rpp::deref(rppHandle));
        }
    }
    else if(interpolationType == RpptInterpolationType::BILINEAR)
    {
        if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
        {
            remap_bilinear_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                             srcDescPtr,
                                             static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                             dstDescPtr,
                                             rowRemapTable,
                                             colRemapTable,
                                             tableDescPtr,
                                             roiTensorPtrSrc,
                                             roiType,
                                             layoutParams,
                                             rpp::deref(rppHandle));
        }
        else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
        {
            remap_bilinear_f16_f16_host_tensor(reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                               srcDescPtr,
                                               reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                               dstDescPtr,
                                               rowRemapTable,
                                               colRemapTable,
                                               tableDescPtr,
                                               roiTensorPtrSrc,
                                               roiType,
                                               layoutParams,
                                               rpp::deref(rppHandle));
        }
        else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
        {
            remap_bilinear_f32_f32_host_tensor(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                               srcDescPtr,
                                               reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                               dstDescPtr,
                                               rowRemapTable,
                                               colRemapTable,
                                               tableDescPtr,
                                               roiTensorPtrSrc,
                                               roiType,
                                               layoutParams,
                                               rpp::deref(rppHandle));
        }
        else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
        {
            remap_bilinear_i8_i8_host_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                             srcDescPtr,
                                             static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                             dstDescPtr,
                                             rowRemapTable,
                                             colRemapTable,
                                             tableDescPtr,
                                             roiTensorPtrSrc,
                                             roiType,
                                             layoutParams,
                                             rpp::deref(rppHandle));
        }
    }

    return RPP_SUCCESS;
}


/********************************************************************************************************************/
/*********************************************** RPP_GPU_SUPPORT = ON ***********************************************/
/********************************************************************************************************************/

#ifdef GPU_SUPPORT

/******************** crop ********************/

RppStatus rppt_crop_gpu(RppPtr_t srcPtr,
                        RpptDescPtr srcDescPtr,
                        RppPtr_t dstPtr,
                        RpptDescPtr dstDescPtr,
                        RpptROIPtr roiTensorPtrSrc,
                        RpptRoiType roiType,
                        rppHandle_t rppHandle)
{
#ifdef HIP_COMPILE
    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        hip_exec_crop_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                             srcDescPtr,
                             static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                             dstDescPtr,
                             roiTensorPtrSrc,
                             roiType,
                             rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        hip_exec_crop_tensor((half*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                             srcDescPtr,
                             (half*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                             dstDescPtr,
                             roiTensorPtrSrc,
                             roiType,
                             rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        hip_exec_crop_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                             srcDescPtr,
                             (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                             dstDescPtr,
                             roiTensorPtrSrc,
                             roiType,
                             rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        hip_exec_crop_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                             srcDescPtr,
                             static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                             dstDescPtr,
                             roiTensorPtrSrc,
                             roiType,
                             rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
#elif defined(OCL_COMPILE)
    return RPP_ERROR_NOT_IMPLEMENTED;
#endif // backend
}

/******************** crop mirror normalize ********************/

RppStatus rppt_crop_mirror_normalize_gpu(RppPtr_t srcPtr,
                                         RpptDescPtr srcDescPtr,
                                         RppPtr_t dstPtr,
                                         RpptDescPtr dstDescPtr,
                                         Rpp32f *offsetTensor,
                                         Rpp32f *multiplierTensor,
                                         Rpp32u *mirrorTensor,
                                         RpptROIPtr roiTensorPtrSrc,
                                         RpptRoiType roiType,
                                         rppHandle_t rppHandle)
{
#ifdef HIP_COMPILE
    Rpp32u paramIndex = 0;
    if(srcDescPtr->c == 3)
    {
        copy_param_float3(offsetTensor, rpp::deref(rppHandle), paramIndex++);
        copy_param_float3(multiplierTensor, rpp::deref(rppHandle), paramIndex++);
    }
    else if(srcDescPtr->c == 1)
    {
        copy_param_float(offsetTensor, rpp::deref(rppHandle), paramIndex++);
        copy_param_float(multiplierTensor, rpp::deref(rppHandle), paramIndex++);
    }
    copy_param_uint(mirrorTensor, rpp::deref(rppHandle), paramIndex++);

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        hip_exec_crop_mirror_normalize_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                              srcDescPtr,
                                              static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                              dstDescPtr,
                                              roiTensorPtrSrc,
                                              roiType,
                                              rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        hip_exec_crop_mirror_normalize_tensor((half*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                              srcDescPtr,
                                              (half*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                              dstDescPtr,
                                              roiTensorPtrSrc,
                                              roiType,
                                              rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        hip_exec_crop_mirror_normalize_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                              srcDescPtr,
                                              (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                              dstDescPtr,
                                              roiTensorPtrSrc,
                                              roiType,
                                              rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        hip_exec_crop_mirror_normalize_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                              srcDescPtr,
                                              static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                              dstDescPtr,
                                              roiTensorPtrSrc,
                                              roiType,
                                              rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        hip_exec_crop_mirror_normalize_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                              srcDescPtr,
                                              (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                              dstDescPtr,
                                              roiTensorPtrSrc,
                                              roiType,
                                              rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        hip_exec_crop_mirror_normalize_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                              srcDescPtr,
                                              (half*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                              dstDescPtr,
                                              roiTensorPtrSrc,
                                              roiType,
                                              rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
#elif defined(OCL_COMPILE)
    return RPP_ERROR_NOT_IMPLEMENTED;
#endif // backend
}

/******************** warp_affine ********************/

RppStatus rppt_warp_affine_gpu(RppPtr_t srcPtr,
                               RpptDescPtr srcDescPtr,
                               RppPtr_t dstPtr,
                               RpptDescPtr dstDescPtr,
                               Rpp32f *affineTensor,
                               RpptInterpolationType interpolationType,
                               RpptROIPtr roiTensorPtrSrc,
                               RpptRoiType roiType,
                               rppHandle_t rppHandle)
{
#ifdef HIP_COMPILE
    if ((interpolationType != RpptInterpolationType::BILINEAR) && (interpolationType != RpptInterpolationType::NEAREST_NEIGHBOR))
        return RPP_ERROR_NOT_IMPLEMENTED;

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        hip_exec_warp_affine_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                    srcDescPtr,
                                    static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                    dstDescPtr,
                                    affineTensor,
                                    interpolationType,
                                    roiTensorPtrSrc,
                                    roiType,
                                    rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        hip_exec_warp_affine_tensor((half*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                    srcDescPtr,
                                    (half*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                    dstDescPtr,
                                    affineTensor,
                                    interpolationType,
                                    roiTensorPtrSrc,
                                    roiType,
                                    rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        hip_exec_warp_affine_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                    srcDescPtr,
                                    (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                    dstDescPtr,
                                    affineTensor,
                                    interpolationType,
                                    roiTensorPtrSrc,
                                    roiType,
                                    rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        hip_exec_warp_affine_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                    srcDescPtr,
                                    static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                    dstDescPtr,
                                    affineTensor,
                                    interpolationType,
                                    roiTensorPtrSrc,
                                    roiType,
                                    rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
#elif defined(OCL_COMPILE)
    return RPP_ERROR_NOT_IMPLEMENTED;
#endif // backend
}

NppStatus nppiWarpAffine_8u_C3R(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
				Npp8u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3], int eInterpolation)
{
        int noOfImages = 1;
        int ip_channel = 3;
        RpptDesc srcDesc, dstDesc;
        RpptDescPtr srcDescPtr, dstDescPtr;
        srcDescPtr = &srcDesc;
        dstDescPtr = &dstDesc;

        srcDescPtr->layout = RpptLayout::NHWC;//pkd
        dstDescPtr->layout = RpptLayout::NHWC;
        srcDescPtr->dataType = RpptDataType::U8;
        dstDescPtr->dataType = RpptDataType::U8;

        srcDescPtr->numDims = 4;
        dstDescPtr->numDims = 4;

        srcDescPtr->offsetInBytes = 64;
        dstDescPtr->offsetInBytes = 0;

        srcDescPtr->n = noOfImages;
        srcDescPtr->h = oSrcSize.height;
        srcDescPtr->w = oSrcSize.width;
        srcDescPtr->c = ip_channel;

        dstDescPtr->n = noOfImages;
        dstDescPtr->h = oSrcSize.height;
        dstDescPtr->w = oSrcSize.width;
        dstDescPtr->c = ip_channel;

    // Optionally set w stride as a multiple of 8 for src/dst

        srcDescPtr->w = ((srcDescPtr->w / 8) * 8) + 8;
        dstDescPtr->w = ((dstDescPtr->w / 8) * 8) + 8;

    // Set n/c/h/w strides for src/dst

        srcDescPtr->strides.nStride = srcDescPtr->c * srcDescPtr->w * srcDescPtr->h;
        srcDescPtr->strides.hStride = srcDescPtr->c * srcDescPtr->w;
        srcDescPtr->strides.wStride = srcDescPtr->c;
        srcDescPtr->strides.cStride = 1;

        dstDescPtr->strides.nStride = dstDescPtr->c * dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.hStride = dstDescPtr->c * dstDescPtr->w;
        dstDescPtr->strides.wStride = dstDescPtr->c;
        dstDescPtr->strides.cStride = 1;

	    RpptROI *roiTensorPtrSrc;
	    hipHostMalloc(&roiTensorPtrSrc, noOfImages * sizeof(RpptROI));

	    RpptRoiType roiTypeSrc;
	    roiTypeSrc = RpptRoiType::XYWH;

	    roiTensorPtrSrc[0].xywhROI.xy.x = 0;
	    roiTensorPtrSrc[0].xywhROI.xy.y = 0;
	    roiTensorPtrSrc[0].xywhROI.roiWidth = oSrcSize.width;
	    roiTensorPtrSrc[0].xywhROI.roiHeight = oSrcSize.height;	

	    Rpp32f6 affineTensor_f6[1];
	    Rpp32f *affineTensor = (Rpp32f *)affineTensor_f6;
	    affineTensor_f6[0].data[0] = aCoeffs[0][0];
        affineTensor_f6[0].data[1] = aCoeffs[0][1];
        affineTensor_f6[0].data[2] = aCoeffs[0][2];
        affineTensor_f6[0].data[3] = aCoeffs[1][0];
        affineTensor_f6[0].data[4] = aCoeffs[1][1];
        affineTensor_f6[0].data[5] = aCoeffs[1][2];
		
	    RpptInterpolationType interpolationType;
		
	    if(eInterpolation == NPPI_INTER_NN)
	    {
		    interpolationType = RpptInterpolationType::NEAREST_NEIGHBOR;
	    }else if(eInterpolation == NPPI_INTER_LINEAR)
	    {   
		    interpolationType = RpptInterpolationType::BILINEAR;
	    }

        unsigned long long ioBufferSize = 0;
        unsigned long long oBufferSize = 0;

        ioBufferSize = (unsigned long long)srcDescPtr->h * (unsigned long long)srcDescPtr->w * (unsigned long long)srcDescPtr->c * (unsigned long long)noOfImages;
        oBufferSize = (unsigned long long)dstDescPtr->h * (unsigned long long)dstDescPtr->w * (unsigned long long)dstDescPtr->c * (unsigned long long)noOfImages;

        unsigned long long ioBufferSizeInBytes_u8 = ioBufferSize + srcDescPtr->offsetInBytes;
        unsigned long long oBufferSizeInBytes_u8 = oBufferSize + dstDescPtr->offsetInBytes;

        Rpp8u *d_input,*d_output;
	    hipMalloc(&d_input,ioBufferSizeInBytes_u8);
        hipMalloc(&d_output,oBufferSizeInBytes_u8);
        Rpp8u *offsetted_input, *offsetted_output;
        offsetted_input = d_input + srcDescPtr->offsetInBytes;
        Rpp8u *temp_in = (Rpp8u *)pSrc;

        Rpp32u elementsInRow = oSrcSize.width * srcDescPtr->c;

        for (int j = 0; j < oSrcSize.height; j++)
        {
            hipMemcpy(offsetted_input, temp_in, elementsInRow * sizeof (Rpp8u), hipMemcpyDeviceToDevice);
            temp_in += elementsInRow;
            offsetted_input += srcDescPtr->strides.hStride;
        }
		
        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        status = rppt_warp_affine_gpu(d_input, srcDescPtr, d_output, dstDescPtr, affineTensor, interpolationType, roiTensorPtrSrc, roiTypeSrc, handle);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        Rpp32u elementsInRowMax = dstDescPtr->w * dstDescPtr->c;
        Rpp32u elementsInRowout = oSrcSize.width * dstDescPtr->c;
        offsetted_output = d_output + dstDescPtr->offsetInBytes;
        Rpp8u *temp_output = pDst;
        for (int k = 0; k < oSrcSize.height; k++)
        {
            hipMemcpy(temp_output, offsetted_output, elementsInRowout * sizeof (Rpp8u),hipMemcpyDeviceToDevice);
            temp_output += elementsInRowout;
            offsetted_output += elementsInRowMax;
	    }

	    hipFree(d_input);
        hipFree(d_output);
	    hipHostFree(roiTensorPtrSrc);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiWarpAffine_8u_C1R_Ctx(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp8u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3], int eInterpolation, NppStreamContext nppStreamCtx)
{
        int noOfImages = 1;
        int ip_channel = 1;
        RpptDesc srcDesc, dstDesc;
        RpptDescPtr srcDescPtr, dstDescPtr;
        srcDescPtr = &srcDesc;
        dstDescPtr = &dstDesc;

        srcDescPtr->layout = RpptLayout::NCHW;//pln
        dstDescPtr->layout = RpptLayout::NCHW;
        srcDescPtr->dataType = RpptDataType::U8;
        dstDescPtr->dataType = RpptDataType::U8;

        srcDescPtr->numDims = 4;
        dstDescPtr->numDims = 4;

        srcDescPtr->offsetInBytes = 64;
        dstDescPtr->offsetInBytes = 0;

        srcDescPtr->n = noOfImages;
        srcDescPtr->h = oSrcSize.height;
        srcDescPtr->w = oSrcSize.width;
        srcDescPtr->c = ip_channel;

        dstDescPtr->n = noOfImages;
        dstDescPtr->h = oSrcSize.height;
        dstDescPtr->w = oSrcSize.width;
        dstDescPtr->c = ip_channel;

    // Optionally set w stride as a multiple of 8 for src/dst

        srcDescPtr->w = ((srcDescPtr->w / 8) * 8) + 8;
        dstDescPtr->w = ((dstDescPtr->w / 8) * 8) + 8;

    // Set n/c/h/w strides for src/dst

        srcDescPtr->strides.nStride = ip_channel * srcDescPtr->w * srcDescPtr->h;
	    srcDescPtr->strides.cStride = srcDescPtr->w * srcDescPtr->h;
	    srcDescPtr->strides.hStride = srcDescPtr->w;
	    srcDescPtr->strides.wStride = 1;

        dstDescPtr->strides.nStride = ip_channel * dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.cStride = dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.hStride = dstDescPtr->w;
        dstDescPtr->strides.wStride = 1;

        RpptROI *roiTensorPtrSrc;
        hipHostMalloc(&roiTensorPtrSrc, noOfImages * sizeof(RpptROI));

        RpptRoiType roiTypeSrc;
        roiTypeSrc = RpptRoiType::XYWH;
		
	    roiTensorPtrSrc[0].xywhROI.xy.x = 0;
        roiTensorPtrSrc[0].xywhROI.xy.y = 0;
        roiTensorPtrSrc[0].xywhROI.roiWidth = oSrcSize.width;
        roiTensorPtrSrc[0].xywhROI.roiHeight = oSrcSize.height;

        Rpp32f6 affineTensor_f6[1];
        Rpp32f *affineTensor = (Rpp32f *)affineTensor_f6;
        affineTensor_f6[0].data[0] = aCoeffs[0][0];
        affineTensor_f6[0].data[1] = aCoeffs[0][1];
        affineTensor_f6[0].data[2] = aCoeffs[0][2];
        affineTensor_f6[0].data[3] = aCoeffs[1][0];
        affineTensor_f6[0].data[4] = aCoeffs[1][1];
        affineTensor_f6[0].data[5] = aCoeffs[1][2];

        RpptInterpolationType interpolationType;

        if(eInterpolation == NPPI_INTER_NN)
        {
            interpolationType = RpptInterpolationType::NEAREST_NEIGHBOR;
        }else if(eInterpolation == NPPI_INTER_LINEAR)
        {
            interpolationType = RpptInterpolationType::BILINEAR;
        }

        unsigned long long ioBufferSize = 0;
        unsigned long long oBufferSize = 0;

        ioBufferSize = (unsigned long long)srcDescPtr->h * (unsigned long long)srcDescPtr->w * (unsigned long long)srcDescPtr->c * (unsigned long long)noOfImages;
        oBufferSize = (unsigned long long)dstDescPtr->h * (unsigned long long)dstDescPtr->w * (unsigned long long)dstDescPtr->c * (unsigned long long)noOfImages;

        unsigned long long ioBufferSizeInBytes_u8 = ioBufferSize + srcDescPtr->offsetInBytes;
        unsigned long long oBufferSizeInBytes_u8 = oBufferSize + dstDescPtr->offsetInBytes;

        Rpp8u *d_input,*d_output;
        hipMalloc(&d_input,ioBufferSizeInBytes_u8);
        hipMalloc(&d_output,oBufferSizeInBytes_u8);
        Rpp8u *offsetted_input, *offsetted_output;
        offsetted_input = d_input + srcDescPtr->offsetInBytes;
        Rpp8u *temp_in = (Rpp8u *)pSrc;

        Rpp32u elementsInRow = oSrcSize.width * srcDescPtr->c;

        for (int j = 0; j < oSrcSize.height; j++)
        {
            hipMemcpy(offsetted_input, temp_in, elementsInRow * sizeof (Rpp8u), hipMemcpyDeviceToDevice);
            temp_in += elementsInRow;
            offsetted_input += srcDescPtr->strides.hStride;
        }

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
	    rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        status = rppt_warp_affine_gpu(d_input, srcDescPtr, d_output, dstDescPtr, affineTensor, interpolationType, roiTensorPtrSrc, roiTypeSrc, handle);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        Rpp32u elementsInRowMax = dstDescPtr->w * dstDescPtr->c;
        Rpp32u elementsInRowout = oSrcSize.width * dstDescPtr->c;
        offsetted_output = d_output + dstDescPtr->offsetInBytes;
        Rpp8u *temp_output = pDst;
        for (int k = 0; k < oSrcSize.height; k++)
        {
            hipMemcpy(temp_output, offsetted_output, elementsInRowout * sizeof (Rpp8u),hipMemcpyDeviceToDevice);
            temp_output += elementsInRowout;
            offsetted_output += elementsInRowMax;
        }

        hipFree(d_input);
        hipFree(d_output);
        hipHostFree(roiTensorPtrSrc);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiWarpAffine_8u_C1R(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp8u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3], int eInterpolation)
{
        int noOfImages = 1;
        int ip_channel = 1;
        RpptDesc srcDesc, dstDesc;
        RpptDescPtr srcDescPtr, dstDescPtr;
        srcDescPtr = &srcDesc;
        dstDescPtr = &dstDesc;

        srcDescPtr->layout = RpptLayout::NCHW;//pln
        dstDescPtr->layout = RpptLayout::NCHW;
        srcDescPtr->dataType = RpptDataType::U8;
        dstDescPtr->dataType = RpptDataType::U8;

        srcDescPtr->numDims = 4;
        dstDescPtr->numDims = 4;

        srcDescPtr->offsetInBytes = 64;
        dstDescPtr->offsetInBytes = 0;

        srcDescPtr->n = noOfImages;
        srcDescPtr->h = oSrcSize.height;
        srcDescPtr->w = oSrcSize.width;
        srcDescPtr->c = ip_channel;

        dstDescPtr->n = noOfImages;
        dstDescPtr->h = oSrcSize.height;
        dstDescPtr->w = oSrcSize.width;
        dstDescPtr->c = ip_channel;

    // Optionally set w stride as a multiple of 8 for src/dst

        srcDescPtr->w = ((srcDescPtr->w / 8) * 8) + 8;
        dstDescPtr->w = ((dstDescPtr->w / 8) * 8) + 8;

    // Set n/c/h/w strides for src/dst

        srcDescPtr->strides.nStride = ip_channel * srcDescPtr->w * srcDescPtr->h;
	    srcDescPtr->strides.cStride = srcDescPtr->w * srcDescPtr->h;
	    srcDescPtr->strides.hStride = srcDescPtr->w;
	    srcDescPtr->strides.wStride = 1;

        dstDescPtr->strides.nStride = ip_channel * dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.cStride = dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.hStride = dstDescPtr->w;
        dstDescPtr->strides.wStride = 1;

        RpptROI *roiTensorPtrSrc;
        hipHostMalloc(&roiTensorPtrSrc, noOfImages * sizeof(RpptROI));

        RpptRoiType roiTypeSrc;
        roiTypeSrc = RpptRoiType::XYWH;
		
	    roiTensorPtrSrc[0].xywhROI.xy.x = 0;
        roiTensorPtrSrc[0].xywhROI.xy.y = 0;
        roiTensorPtrSrc[0].xywhROI.roiWidth = oSrcSize.width;
        roiTensorPtrSrc[0].xywhROI.roiHeight = oSrcSize.height;

        Rpp32f6 affineTensor_f6[1];
        Rpp32f *affineTensor = (Rpp32f *)affineTensor_f6;
        affineTensor_f6[0].data[0] = aCoeffs[0][0];
        affineTensor_f6[0].data[1] = aCoeffs[0][1];
        affineTensor_f6[0].data[2] = aCoeffs[0][2];
        affineTensor_f6[0].data[3] = aCoeffs[1][0];
        affineTensor_f6[0].data[4] = aCoeffs[1][1];
        affineTensor_f6[0].data[5] = aCoeffs[1][2];

        RpptInterpolationType interpolationType;

        if(eInterpolation == NPPI_INTER_NN)
        {
            interpolationType = RpptInterpolationType::NEAREST_NEIGHBOR;
        }else if(eInterpolation == NPPI_INTER_LINEAR)
        {
            interpolationType = RpptInterpolationType::BILINEAR;
        }

        unsigned long long ioBufferSize = 0;
        unsigned long long oBufferSize = 0;

        ioBufferSize = (unsigned long long)srcDescPtr->h * (unsigned long long)srcDescPtr->w * (unsigned long long)srcDescPtr->c * (unsigned long long)noOfImages;
        oBufferSize = (unsigned long long)dstDescPtr->h * (unsigned long long)dstDescPtr->w * (unsigned long long)dstDescPtr->c * (unsigned long long)noOfImages;

        unsigned long long ioBufferSizeInBytes_u8 = ioBufferSize + srcDescPtr->offsetInBytes;
        unsigned long long oBufferSizeInBytes_u8 = oBufferSize + dstDescPtr->offsetInBytes;

        Rpp8u *d_input,*d_output;
        hipMalloc(&d_input,ioBufferSizeInBytes_u8);
        hipMalloc(&d_output,oBufferSizeInBytes_u8);
        Rpp8u *offsetted_input, *offsetted_output;
        offsetted_input = d_input + srcDescPtr->offsetInBytes;
        Rpp8u *temp_in = (Rpp8u *)pSrc;

        Rpp32u elementsInRow = oSrcSize.width * srcDescPtr->c;

        for (int j = 0; j < oSrcSize.height; j++)
        {
            hipMemcpy(offsetted_input, temp_in, elementsInRow * sizeof (Rpp8u), hipMemcpyDeviceToDevice);
            temp_in += elementsInRow;
            offsetted_input += srcDescPtr->strides.hStride;
        }

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
	    rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        status = rppt_warp_affine_gpu(d_input, srcDescPtr, d_output, dstDescPtr, affineTensor, interpolationType, roiTensorPtrSrc, roiTypeSrc, handle);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        Rpp32u elementsInRowMax = dstDescPtr->w * dstDescPtr->c;
        Rpp32u elementsInRowout = oSrcSize.width * dstDescPtr->c;
        offsetted_output = d_output + dstDescPtr->offsetInBytes;
        Rpp8u *temp_output = pDst;
        for (int k = 0; k < oSrcSize.height; k++)
        {
            hipMemcpy(temp_output, offsetted_output, elementsInRowout * sizeof (Rpp8u),hipMemcpyDeviceToDevice);
            temp_output += elementsInRowout;
            offsetted_output += elementsInRowMax;
        }

        hipFree(d_input);
        hipFree(d_output);
        hipHostFree(roiTensorPtrSrc);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiWarpAffine_8u_C3R_Ctx(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp8u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3], int eInterpolation, NppStreamContext nppStreamCtx)
{
        int noOfImages = 1;
        int ip_channel = 3;
        RpptDesc srcDesc, dstDesc;
        RpptDescPtr srcDescPtr, dstDescPtr;
        srcDescPtr = &srcDesc;
        dstDescPtr = &dstDesc;

        srcDescPtr->layout = RpptLayout::NHWC;//pkd
        dstDescPtr->layout = RpptLayout::NHWC;
        srcDescPtr->dataType = RpptDataType::U8;
        dstDescPtr->dataType = RpptDataType::U8;

        srcDescPtr->numDims = 4;
        dstDescPtr->numDims = 4;

        srcDescPtr->offsetInBytes = 64;
        dstDescPtr->offsetInBytes = 0;

        srcDescPtr->n = noOfImages;
        srcDescPtr->h = oSrcSize.height;
        srcDescPtr->w = oSrcSize.width;
        srcDescPtr->c = ip_channel;

        dstDescPtr->n = noOfImages;
        dstDescPtr->h = oSrcSize.height;
        dstDescPtr->w = oSrcSize.width;
        dstDescPtr->c = ip_channel;

    // Optionally set w stride as a multiple of 8 for src/dst

        srcDescPtr->w = ((srcDescPtr->w / 8) * 8) + 8;
        dstDescPtr->w = ((dstDescPtr->w / 8) * 8) + 8;

    // Set n/c/h/w strides for src/dst

        srcDescPtr->strides.nStride = srcDescPtr->c * srcDescPtr->w * srcDescPtr->h;
        srcDescPtr->strides.hStride = srcDescPtr->c * srcDescPtr->w;
        srcDescPtr->strides.wStride = srcDescPtr->c;
        srcDescPtr->strides.cStride = 1;

        dstDescPtr->strides.nStride = dstDescPtr->c * dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.hStride = dstDescPtr->c * dstDescPtr->w;
        dstDescPtr->strides.wStride = dstDescPtr->c;
        dstDescPtr->strides.cStride = 1;

        RpptROI *roiTensorPtrSrc;
        hipHostMalloc(&roiTensorPtrSrc, noOfImages * sizeof(RpptROI));

        RpptRoiType roiTypeSrc;
        roiTypeSrc = RpptRoiType::XYWH;
		
	    roiTensorPtrSrc[0].xywhROI.xy.x = 0;
        roiTensorPtrSrc[0].xywhROI.xy.y = 0;
        roiTensorPtrSrc[0].xywhROI.roiWidth = oSrcSize.width;
        roiTensorPtrSrc[0].xywhROI.roiHeight = oSrcSize.height;

        Rpp32f6 affineTensor_f6[1];
        Rpp32f *affineTensor = (Rpp32f *)affineTensor_f6;
        affineTensor_f6[0].data[0] = aCoeffs[0][0];
        affineTensor_f6[0].data[1] = aCoeffs[0][1];
        affineTensor_f6[0].data[2] = aCoeffs[0][2];
        affineTensor_f6[0].data[3] = aCoeffs[1][0];
        affineTensor_f6[0].data[4] = aCoeffs[1][1];
        affineTensor_f6[0].data[5] = aCoeffs[1][2];

        RpptInterpolationType interpolationType;

        if(eInterpolation == NPPI_INTER_NN)
        {
                interpolationType = RpptInterpolationType::NEAREST_NEIGHBOR;
        }else if(eInterpolation == NPPI_INTER_LINEAR)
        {
                interpolationType = RpptInterpolationType::BILINEAR;
        }

        unsigned long long ioBufferSize = 0;
        unsigned long long oBufferSize = 0;

        ioBufferSize = (unsigned long long)srcDescPtr->h * (unsigned long long)srcDescPtr->w * (unsigned long long)srcDescPtr->c * (unsigned long long)noOfImages;
        oBufferSize = (unsigned long long)dstDescPtr->h * (unsigned long long)dstDescPtr->w * (unsigned long long)dstDescPtr->c * (unsigned long long)noOfImages;

        unsigned long long ioBufferSizeInBytes_u8 = ioBufferSize + srcDescPtr->offsetInBytes;
        unsigned long long oBufferSizeInBytes_u8 = oBufferSize + dstDescPtr->offsetInBytes;

        Rpp8u *d_input,*d_output;
        hipMalloc(&d_input,ioBufferSizeInBytes_u8);
        hipMalloc(&d_output,oBufferSizeInBytes_u8);
        Rpp8u *offsetted_input, *offsetted_output;
        offsetted_input = d_input + srcDescPtr->offsetInBytes;
        Rpp8u *temp_in = (Rpp8u *)pSrc;

        Rpp32u elementsInRow = oSrcSize.width * srcDescPtr->c;

        for (int j = 0; j < oSrcSize.height; j++)
        {
            hipMemcpy(offsetted_input, temp_in, elementsInRow * sizeof (Rpp8u), hipMemcpyDeviceToDevice);
            temp_in += elementsInRow;
            offsetted_input += srcDescPtr->strides.hStride;
        }

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
	    rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        status = rppt_warp_affine_gpu(d_input, srcDescPtr, d_output, dstDescPtr, affineTensor, interpolationType, roiTensorPtrSrc, roiTypeSrc, handle);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        Rpp32u elementsInRowMax = dstDescPtr->w * dstDescPtr->c;
        Rpp32u elementsInRowout = oSrcSize.width * dstDescPtr->c;
        offsetted_output = d_output + dstDescPtr->offsetInBytes;
        Rpp8u *temp_output = pDst;
        for (int k = 0; k < oSrcSize.height; k++)
        {
            hipMemcpy(temp_output, offsetted_output, elementsInRowout * sizeof (Rpp8u),hipMemcpyDeviceToDevice);
            temp_output += elementsInRowout;
            offsetted_output += elementsInRowMax;
        }

        hipFree(d_input);
        hipFree(d_output);
        hipHostFree(roiTensorPtrSrc);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiWarpAffine_16f_C1R_Ctx(const Npp16f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp16f *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3], int eInterpolation, NppStreamContext nppStreamCtx)
{
        int noOfImages = 1;
        int ip_channel = 1;
        RpptDesc srcDesc, dstDesc;
        RpptDescPtr srcDescPtr, dstDescPtr;
        srcDescPtr = &srcDesc;
        dstDescPtr = &dstDesc;

        srcDescPtr->layout = RpptLayout::NCHW;//pln
        dstDescPtr->layout = RpptLayout::NCHW;
        srcDescPtr->dataType = RpptDataType::F16;
        dstDescPtr->dataType = RpptDataType::F16;

        srcDescPtr->numDims = 4;
        dstDescPtr->numDims = 4;

        srcDescPtr->offsetInBytes = 64;
        dstDescPtr->offsetInBytes = 0;

        srcDescPtr->n = noOfImages;
        srcDescPtr->h = oSrcSize.height;
        srcDescPtr->w = oSrcSize.width;
        srcDescPtr->c = ip_channel;

        dstDescPtr->n = noOfImages;
        dstDescPtr->h = oSrcSize.height;
        dstDescPtr->w = oSrcSize.width;
        dstDescPtr->c = ip_channel;

    // Optionally set w stride as a multiple of 8 for src/dst

        srcDescPtr->w = ((srcDescPtr->w / 8) * 8) + 8;
        dstDescPtr->w = ((dstDescPtr->w / 8) * 8) + 8;

    // Set n/c/h/w strides for src/dst

        srcDescPtr->strides.nStride = ip_channel * srcDescPtr->w * srcDescPtr->h;
        srcDescPtr->strides.cStride = srcDescPtr->w * srcDescPtr->h;
        srcDescPtr->strides.hStride = srcDescPtr->w;
        srcDescPtr->strides.wStride = 1;

        dstDescPtr->strides.nStride = ip_channel * dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.cStride = dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.hStride = dstDescPtr->w;
        dstDescPtr->strides.wStride = 1;

        RpptROI *roiTensorPtrSrc;
        hipHostMalloc(&roiTensorPtrSrc, noOfImages * sizeof(RpptROI));

        RpptRoiType roiTypeSrc;
        roiTypeSrc = RpptRoiType::XYWH;

        roiTensorPtrSrc[0].xywhROI.xy.x = 0;
        roiTensorPtrSrc[0].xywhROI.xy.y = 0;
        roiTensorPtrSrc[0].xywhROI.roiWidth = oSrcSize.width;
        roiTensorPtrSrc[0].xywhROI.roiHeight = oSrcSize.height;

        Rpp32f6 affineTensor_f6[1];
        Rpp32f *affineTensor = (Rpp32f *)affineTensor_f6;
        affineTensor_f6[0].data[0] = aCoeffs[0][0];
        affineTensor_f6[0].data[1] = aCoeffs[0][1];
        affineTensor_f6[0].data[2] = aCoeffs[0][2];
        affineTensor_f6[0].data[3] = aCoeffs[1][0];
        affineTensor_f6[0].data[4] = aCoeffs[1][1];
        affineTensor_f6[0].data[5] = aCoeffs[1][2];

        RpptInterpolationType interpolationType;

        if(eInterpolation == NPPI_INTER_NN)
        {
            interpolationType = RpptInterpolationType::NEAREST_NEIGHBOR;
        }else if(eInterpolation == NPPI_INTER_LINEAR)
        {
            interpolationType = RpptInterpolationType::BILINEAR;
        }

        unsigned long long ioBufferSize = 0;
        unsigned long long oBufferSize = 0;

        ioBufferSize = (unsigned long long)srcDescPtr->h * (unsigned long long)srcDescPtr->w * (unsigned long long)srcDescPtr->c * (unsigned long long)noOfImages;
        oBufferSize = (unsigned long long)dstDescPtr->h * (unsigned long long)dstDescPtr->w * (unsigned long long)dstDescPtr->c * (unsigned long long)noOfImages;

        unsigned long long ioBufferSizeInBytes_f16 = (ioBufferSize * 2) + srcDescPtr->offsetInBytes;
        unsigned long long oBufferSizeInBytes_f16 = (oBufferSize * 2)+ dstDescPtr->offsetInBytes;

        Rpp16f *d_input,*d_output;
        hipMalloc(&d_input,ioBufferSizeInBytes_f16);
        hipMalloc(&d_output,oBufferSizeInBytes_f16);
        Rpp16f *offsetted_input, *offsetted_output;
        offsetted_input = d_input + srcDescPtr->offsetInBytes;
        Rpp16f *temp_in = (Rpp16f *)pSrc;

        Rpp32u elementsInRow = oSrcSize.width * srcDescPtr->c;

        for (int j = 0; j < oSrcSize.height; j++)
        {
            hipMemcpy(offsetted_input, temp_in, elementsInRow * sizeof (Rpp16f), hipMemcpyDeviceToDevice);
            temp_in += elementsInRow;
            offsetted_input += srcDescPtr->strides.hStride;
        }

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        status = rppt_warp_affine_gpu(d_input, srcDescPtr, d_output, dstDescPtr, affineTensor, interpolationType, roiTensorPtrSrc, roiTypeSrc, handle);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        Rpp32u elementsInRowMax = dstDescPtr->w * dstDescPtr->c;
        Rpp32u elementsInRowout = oSrcSize.width * dstDescPtr->c;
        offsetted_output = d_output + dstDescPtr->offsetInBytes;
        Rpp16f *temp_output = (Rpp16f *)pDst;
        for (int k = 0; k < oSrcSize.height; k++)
        {
            hipMemcpy(temp_output, offsetted_output, elementsInRowout * sizeof (Rpp16f),hipMemcpyDeviceToDevice);
            temp_output += elementsInRowout;
            offsetted_output += elementsInRowMax;
        }

        hipFree(d_input);
        hipFree(d_output);
        hipHostFree(roiTensorPtrSrc);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiWarpAffine_16f_C1R(const Npp16f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp16f *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3], int eInterpolation)
{
        int noOfImages = 1;
        int ip_channel = 1;
        RpptDesc srcDesc, dstDesc;
        RpptDescPtr srcDescPtr, dstDescPtr;
        srcDescPtr = &srcDesc;
        dstDescPtr = &dstDesc;

        srcDescPtr->layout = RpptLayout::NCHW;//pln
        dstDescPtr->layout = RpptLayout::NCHW;
        srcDescPtr->dataType = RpptDataType::F16;
        dstDescPtr->dataType = RpptDataType::F16;

        srcDescPtr->numDims = 4;
        dstDescPtr->numDims = 4;

        srcDescPtr->offsetInBytes = 64;
        dstDescPtr->offsetInBytes = 0;

        srcDescPtr->n = noOfImages;
        srcDescPtr->h = oSrcSize.height;
        srcDescPtr->w = oSrcSize.width;
        srcDescPtr->c = ip_channel;

        dstDescPtr->n = noOfImages;
        dstDescPtr->h = oSrcSize.height;
        dstDescPtr->w = oSrcSize.width;
        dstDescPtr->c = ip_channel;

    // Optionally set w stride as a multiple of 8 for src/dst

        srcDescPtr->w = ((srcDescPtr->w / 8) * 8) + 8;
        dstDescPtr->w = ((dstDescPtr->w / 8) * 8) + 8;

    // Set n/c/h/w strides for src/dst

        srcDescPtr->strides.nStride = ip_channel * srcDescPtr->w * srcDescPtr->h;
        srcDescPtr->strides.cStride = srcDescPtr->w * srcDescPtr->h;
        srcDescPtr->strides.hStride = srcDescPtr->w;
        srcDescPtr->strides.wStride = 1;

        dstDescPtr->strides.nStride = ip_channel * dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.cStride = dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.hStride = dstDescPtr->w;
        dstDescPtr->strides.wStride = 1;

        RpptROI *roiTensorPtrSrc;
        hipHostMalloc(&roiTensorPtrSrc, noOfImages * sizeof(RpptROI));

        RpptRoiType roiTypeSrc;
        roiTypeSrc = RpptRoiType::XYWH;

        roiTensorPtrSrc[0].xywhROI.xy.x = 0;
        roiTensorPtrSrc[0].xywhROI.xy.y = 0;
        roiTensorPtrSrc[0].xywhROI.roiWidth = oSrcSize.width;
        roiTensorPtrSrc[0].xywhROI.roiHeight = oSrcSize.height;

        Rpp32f6 affineTensor_f6[1];
        Rpp32f *affineTensor = (Rpp32f *)affineTensor_f6;
        affineTensor_f6[0].data[0] = aCoeffs[0][0];
        affineTensor_f6[0].data[1] = aCoeffs[0][1];
        affineTensor_f6[0].data[2] = aCoeffs[0][2];
        affineTensor_f6[0].data[3] = aCoeffs[1][0];
        affineTensor_f6[0].data[4] = aCoeffs[1][1];
        affineTensor_f6[0].data[5] = aCoeffs[1][2];

        RpptInterpolationType interpolationType;

        if(eInterpolation == NPPI_INTER_NN)
        {
            interpolationType = RpptInterpolationType::NEAREST_NEIGHBOR;
        }else if(eInterpolation == NPPI_INTER_LINEAR)
        {
            interpolationType = RpptInterpolationType::BILINEAR;
        }

        unsigned long long ioBufferSize = 0;
        unsigned long long oBufferSize = 0;

        ioBufferSize = (unsigned long long)srcDescPtr->h * (unsigned long long)srcDescPtr->w * (unsigned long long)srcDescPtr->c * (unsigned long long)noOfImages;
        oBufferSize = (unsigned long long)dstDescPtr->h * (unsigned long long)dstDescPtr->w * (unsigned long long)dstDescPtr->c * (unsigned long long)noOfImages;

        unsigned long long ioBufferSizeInBytes_f16 = (ioBufferSize * 2) + srcDescPtr->offsetInBytes;
        unsigned long long oBufferSizeInBytes_f16 = (oBufferSize * 2)+ dstDescPtr->offsetInBytes;

        Rpp16f *d_input,*d_output;
        hipMalloc(&d_input,ioBufferSizeInBytes_f16);
        hipMalloc(&d_output,oBufferSizeInBytes_f16);
        Rpp16f *offsetted_input, *offsetted_output;
        offsetted_input = d_input + srcDescPtr->offsetInBytes;
        Rpp16f *temp_in = (Rpp16f *)pSrc;

        Rpp32u elementsInRow = oSrcSize.width * srcDescPtr->c;

        for (int j = 0; j < oSrcSize.height; j++)
        {
            hipMemcpy(offsetted_input, temp_in, elementsInRow * sizeof (Rpp16f), hipMemcpyDeviceToDevice);
            temp_in += elementsInRow;
            offsetted_input += srcDescPtr->strides.hStride;
        }

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        status = rppt_warp_affine_gpu(d_input, srcDescPtr, d_output, dstDescPtr, affineTensor, interpolationType, roiTensorPtrSrc, roiTypeSrc, handle);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        Rpp32u elementsInRowMax = dstDescPtr->w * dstDescPtr->c;
        Rpp32u elementsInRowout = oSrcSize.width * dstDescPtr->c;
        offsetted_output = d_output + dstDescPtr->offsetInBytes;
        Rpp16f *temp_output = (Rpp16f *)pDst;
        for (int k = 0; k < oSrcSize.height; k++)
        {
            hipMemcpy(temp_output, offsetted_output, elementsInRowout * sizeof (Rpp16f),hipMemcpyDeviceToDevice);
            temp_output += elementsInRowout;
            offsetted_output += elementsInRowMax;
        }

        hipFree(d_input);
        hipFree(d_output);
        hipHostFree(roiTensorPtrSrc);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiWarpAffine_16f_C3R_Ctx(const Npp16f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp16f *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3], int eInterpolation, NppStreamContext nppStreamCtx)
{
        int noOfImages = 1;
        int ip_channel = 3;
        RpptDesc srcDesc, dstDesc;
        RpptDescPtr srcDescPtr, dstDescPtr;
        srcDescPtr = &srcDesc;
        dstDescPtr = &dstDesc;

        srcDescPtr->layout = RpptLayout::NHWC;//pkd
        dstDescPtr->layout = RpptLayout::NHWC;
        srcDescPtr->dataType = RpptDataType::F16;
        dstDescPtr->dataType = RpptDataType::F16;

        srcDescPtr->numDims = 4;
        dstDescPtr->numDims = 4;

        srcDescPtr->offsetInBytes = 64;
        dstDescPtr->offsetInBytes = 0;

        srcDescPtr->n = noOfImages;
        srcDescPtr->h = oSrcSize.height;
        srcDescPtr->w = oSrcSize.width;
        srcDescPtr->c = ip_channel;

        dstDescPtr->n = noOfImages;
        dstDescPtr->h = oSrcSize.height;
        dstDescPtr->w = oSrcSize.width;
        dstDescPtr->c = ip_channel;

    // Optionally set w stride as a multiple of 8 for src/dst

        srcDescPtr->w = ((srcDescPtr->w / 8) * 8) + 8;
        dstDescPtr->w = ((dstDescPtr->w / 8) * 8) + 8;

    // Set n/c/h/w strides for src/dst

        srcDescPtr->strides.nStride = srcDescPtr->c * srcDescPtr->w * srcDescPtr->h;
        srcDescPtr->strides.hStride = srcDescPtr->c * srcDescPtr->w;
        srcDescPtr->strides.wStride = srcDescPtr->c;
        srcDescPtr->strides.cStride = 1;

        dstDescPtr->strides.nStride = dstDescPtr->c * dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.hStride = dstDescPtr->c * dstDescPtr->w;
        dstDescPtr->strides.wStride = dstDescPtr->c;
        dstDescPtr->strides.cStride = 1;

        RpptROI *roiTensorPtrSrc;
        hipHostMalloc(&roiTensorPtrSrc, noOfImages * sizeof(RpptROI));

        RpptRoiType roiTypeSrc;
        roiTypeSrc = RpptRoiType::XYWH;

        roiTensorPtrSrc[0].xywhROI.xy.x = 0;
        roiTensorPtrSrc[0].xywhROI.xy.y = 0;
        roiTensorPtrSrc[0].xywhROI.roiWidth = oSrcSize.width;
        roiTensorPtrSrc[0].xywhROI.roiHeight = oSrcSize.height;

        Rpp32f6 affineTensor_f6[1];
        Rpp32f *affineTensor = (Rpp32f *)affineTensor_f6;
        affineTensor_f6[0].data[0] = aCoeffs[0][0];
        affineTensor_f6[0].data[1] = aCoeffs[0][1];
        affineTensor_f6[0].data[2] = aCoeffs[0][2];
        affineTensor_f6[0].data[3] = aCoeffs[1][0];
        affineTensor_f6[0].data[4] = aCoeffs[1][1];
        affineTensor_f6[0].data[5] = aCoeffs[1][2];

        RpptInterpolationType interpolationType;

        if(eInterpolation == NPPI_INTER_NN)
        {
            interpolationType = RpptInterpolationType::NEAREST_NEIGHBOR;
        }else if(eInterpolation == NPPI_INTER_LINEAR)
        {
            interpolationType = RpptInterpolationType::BILINEAR;
        }

        unsigned long long ioBufferSize = 0;
	    unsigned long long oBufferSize = 0;

        ioBufferSize = (unsigned long long)srcDescPtr->h * (unsigned long long)srcDescPtr->w * (unsigned long long)srcDescPtr->c * (unsigned long long)noOfImages;
        oBufferSize = (unsigned long long)dstDescPtr->h * (unsigned long long)dstDescPtr->w * (unsigned long long)dstDescPtr->c * (unsigned long long)noOfImages;

        unsigned long long ioBufferSizeInBytes_f16 = (ioBufferSize * 2) + srcDescPtr->offsetInBytes;
        unsigned long long oBufferSizeInBytes_f16 = (oBufferSize * 2)+ dstDescPtr->offsetInBytes;

        Rpp16f *d_input,*d_output;
        hipMalloc(&d_input,ioBufferSizeInBytes_f16);
        hipMalloc(&d_output,oBufferSizeInBytes_f16);
        Rpp16f *offsetted_input, *offsetted_output;
        offsetted_input = d_input + srcDescPtr->offsetInBytes;
        Rpp16f *temp_in = (Rpp16f *)pSrc;

        Rpp32u elementsInRow = oSrcSize.width * srcDescPtr->c;

        for (int j = 0; j < oSrcSize.height; j++)
        {
            hipMemcpy(offsetted_input, temp_in, elementsInRow * sizeof (Rpp16f), hipMemcpyDeviceToDevice);
            temp_in += elementsInRow;
            offsetted_input += srcDescPtr->strides.hStride;
        }

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        status = rppt_warp_affine_gpu(d_input, srcDescPtr, d_output, dstDescPtr, affineTensor, interpolationType, roiTensorPtrSrc, roiTypeSrc, handle);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        Rpp32u elementsInRowMax = dstDescPtr->w * dstDescPtr->c;
        Rpp32u elementsInRowout = oSrcSize.width * dstDescPtr->c;
        offsetted_output = d_output + dstDescPtr->offsetInBytes;
        Rpp16f *temp_output = (Rpp16f *)pDst;
        for (int k = 0; k < oSrcSize.height; k++)
        {
            hipMemcpy(temp_output, offsetted_output, elementsInRowout * sizeof (Rpp16f),hipMemcpyDeviceToDevice);
            temp_output += elementsInRowout;
            offsetted_output += elementsInRowMax;
        }

        hipFree(d_input);
        hipFree(d_output);
        hipHostFree(roiTensorPtrSrc);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiWarpAffine_16f_C3R(const Npp16f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp16f *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3], int eInterpolation)
{
        int noOfImages = 1;
        int ip_channel = 3;
        RpptDesc srcDesc, dstDesc;
        RpptDescPtr srcDescPtr, dstDescPtr;
        srcDescPtr = &srcDesc;
        dstDescPtr = &dstDesc;

        srcDescPtr->layout = RpptLayout::NHWC;//pkd
        dstDescPtr->layout = RpptLayout::NHWC;
        srcDescPtr->dataType = RpptDataType::F16;
        dstDescPtr->dataType = RpptDataType::F16;

        srcDescPtr->numDims = 4;
        dstDescPtr->numDims = 4;

        srcDescPtr->offsetInBytes = 64;
        dstDescPtr->offsetInBytes = 0;

        srcDescPtr->n = noOfImages;
        srcDescPtr->h = oSrcSize.height;
        srcDescPtr->w = oSrcSize.width;
        srcDescPtr->c = ip_channel;

        dstDescPtr->n = noOfImages;
        dstDescPtr->h = oSrcSize.height;
        dstDescPtr->w = oSrcSize.width;
        dstDescPtr->c = ip_channel;

    // Optionally set w stride as a multiple of 8 for src/dst

        srcDescPtr->w = ((srcDescPtr->w / 8) * 8) + 8;
        dstDescPtr->w = ((dstDescPtr->w / 8) * 8) + 8;

    // Set n/c/h/w strides for src/dst

        srcDescPtr->strides.nStride = srcDescPtr->c * srcDescPtr->w * srcDescPtr->h;
        srcDescPtr->strides.hStride = srcDescPtr->c * srcDescPtr->w;
        srcDescPtr->strides.wStride = srcDescPtr->c;
        srcDescPtr->strides.cStride = 1;

        dstDescPtr->strides.nStride = dstDescPtr->c * dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.hStride = dstDescPtr->c * dstDescPtr->w;
        dstDescPtr->strides.wStride = dstDescPtr->c;
        dstDescPtr->strides.cStride = 1;

        RpptROI *roiTensorPtrSrc;
        hipHostMalloc(&roiTensorPtrSrc, noOfImages * sizeof(RpptROI));

        RpptRoiType roiTypeSrc;
        roiTypeSrc = RpptRoiType::XYWH;

        roiTensorPtrSrc[0].xywhROI.xy.x = 0;
        roiTensorPtrSrc[0].xywhROI.xy.y = 0;
        roiTensorPtrSrc[0].xywhROI.roiWidth = oSrcSize.width;
        roiTensorPtrSrc[0].xywhROI.roiHeight = oSrcSize.height;

        Rpp32f6 affineTensor_f6[1];
        Rpp32f *affineTensor = (Rpp32f *)affineTensor_f6;
        affineTensor_f6[0].data[0] = aCoeffs[0][0];
        affineTensor_f6[0].data[1] = aCoeffs[0][1];
        affineTensor_f6[0].data[2] = aCoeffs[0][2];
        affineTensor_f6[0].data[3] = aCoeffs[1][0];
        affineTensor_f6[0].data[4] = aCoeffs[1][1];
        affineTensor_f6[0].data[5] = aCoeffs[1][2];

        RpptInterpolationType interpolationType;

        if(eInterpolation == NPPI_INTER_NN)
        {
            interpolationType = RpptInterpolationType::NEAREST_NEIGHBOR;
        }else if(eInterpolation == NPPI_INTER_LINEAR)
        {
            interpolationType = RpptInterpolationType::BILINEAR;
        }

        unsigned long long ioBufferSize = 0;
	    unsigned long long oBufferSize = 0;

        ioBufferSize = (unsigned long long)srcDescPtr->h * (unsigned long long)srcDescPtr->w * (unsigned long long)srcDescPtr->c * (unsigned long long)noOfImages;
        oBufferSize = (unsigned long long)dstDescPtr->h * (unsigned long long)dstDescPtr->w * (unsigned long long)dstDescPtr->c * (unsigned long long)noOfImages;

        unsigned long long ioBufferSizeInBytes_f16 = (ioBufferSize * 2) + srcDescPtr->offsetInBytes;
        unsigned long long oBufferSizeInBytes_f16 = (oBufferSize * 2)+ dstDescPtr->offsetInBytes;

        Rpp16f *d_input,*d_output;
        hipMalloc(&d_input,ioBufferSizeInBytes_f16);
        hipMalloc(&d_output,oBufferSizeInBytes_f16);
        Rpp16f *offsetted_input, *offsetted_output;
        offsetted_input = d_input + srcDescPtr->offsetInBytes;
        Rpp16f *temp_in = (Rpp16f *)pSrc;

        Rpp32u elementsInRow = oSrcSize.width * srcDescPtr->c;

        for (int j = 0; j < oSrcSize.height; j++)
        {
            hipMemcpy(offsetted_input, temp_in, elementsInRow * sizeof (Rpp16f), hipMemcpyDeviceToDevice);
            temp_in += elementsInRow;
            offsetted_input += srcDescPtr->strides.hStride;
        }

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        status = rppt_warp_affine_gpu(d_input, srcDescPtr, d_output, dstDescPtr, affineTensor, interpolationType, roiTensorPtrSrc, roiTypeSrc, handle);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        Rpp32u elementsInRowMax = dstDescPtr->w * dstDescPtr->c;
        Rpp32u elementsInRowout = oSrcSize.width * dstDescPtr->c;
        offsetted_output = d_output + dstDescPtr->offsetInBytes;
        Rpp16f *temp_output = (Rpp16f *)pDst;
        for (int k = 0; k < oSrcSize.height; k++)
        {
            hipMemcpy(temp_output, offsetted_output, elementsInRowout * sizeof (Rpp16f),hipMemcpyDeviceToDevice);
            temp_output += elementsInRowout;
            offsetted_output += elementsInRowMax;
        }

        hipFree(d_input);
        hipFree(d_output);
        hipHostFree(roiTensorPtrSrc);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiWarpAffine_32f_C1R_Ctx(const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp32f *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3], int eInterpolation, NppStreamContext nppStreamCtx)
{
        int noOfImages = 1;
        int ip_channel = 1;
        RpptDesc srcDesc, dstDesc;
        RpptDescPtr srcDescPtr, dstDescPtr;
        srcDescPtr = &srcDesc;
        dstDescPtr = &dstDesc;

        srcDescPtr->layout = RpptLayout::NCHW;//pln
        dstDescPtr->layout = RpptLayout::NCHW;
        srcDescPtr->dataType = RpptDataType::F32;
        dstDescPtr->dataType = RpptDataType::F32;

        srcDescPtr->numDims = 4;
        dstDescPtr->numDims = 4;

        srcDescPtr->offsetInBytes = 64;
        dstDescPtr->offsetInBytes = 0;

        srcDescPtr->n = noOfImages;
        srcDescPtr->h = oSrcSize.height;
        srcDescPtr->w = oSrcSize.width;
        srcDescPtr->c = ip_channel;

        dstDescPtr->n = noOfImages;
        dstDescPtr->h = oSrcSize.height;
        dstDescPtr->w = oSrcSize.width;
        dstDescPtr->c = ip_channel;

    // Optionally set w stride as a multiple of 8 for src/dst

        srcDescPtr->w = ((srcDescPtr->w / 8) * 8) + 8;
        dstDescPtr->w = ((dstDescPtr->w / 8) * 8) + 8;

    // Set n/c/h/w strides for src/dst

        srcDescPtr->strides.nStride = ip_channel * srcDescPtr->w * srcDescPtr->h;
        srcDescPtr->strides.cStride = srcDescPtr->w * srcDescPtr->h;
        srcDescPtr->strides.hStride = srcDescPtr->w;
        srcDescPtr->strides.wStride = 1;

        dstDescPtr->strides.nStride = ip_channel * dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.cStride = dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.hStride = dstDescPtr->w;
        dstDescPtr->strides.wStride = 1;

        RpptROI *roiTensorPtrSrc;
        hipHostMalloc(&roiTensorPtrSrc, noOfImages * sizeof(RpptROI));

        RpptRoiType roiTypeSrc;
        roiTypeSrc = RpptRoiType::XYWH;

        roiTensorPtrSrc[0].xywhROI.xy.x = 0;
        roiTensorPtrSrc[0].xywhROI.xy.y = 0;
        roiTensorPtrSrc[0].xywhROI.roiWidth = oSrcSize.width;
        roiTensorPtrSrc[0].xywhROI.roiHeight = oSrcSize.height;

        Rpp32f6 affineTensor_f6[1];
        Rpp32f *affineTensor = (Rpp32f *)affineTensor_f6;
        affineTensor_f6[0].data[0] = aCoeffs[0][0];
        affineTensor_f6[0].data[1] = aCoeffs[0][1];
        affineTensor_f6[0].data[2] = aCoeffs[0][2];
        affineTensor_f6[0].data[3] = aCoeffs[1][0];
        affineTensor_f6[0].data[4] = aCoeffs[1][1];
        affineTensor_f6[0].data[5] = aCoeffs[1][2];

        RpptInterpolationType interpolationType;

        if(eInterpolation == NPPI_INTER_NN)
        {
            interpolationType = RpptInterpolationType::NEAREST_NEIGHBOR;
        }else if(eInterpolation == NPPI_INTER_LINEAR)
        {
            interpolationType = RpptInterpolationType::BILINEAR;
        }

        unsigned long long ioBufferSize = 0;
        unsigned long long oBufferSize = 0;

        ioBufferSize = (unsigned long long)srcDescPtr->h * (unsigned long long)srcDescPtr->w * (unsigned long long)srcDescPtr->c * (unsigned long long)noOfImages;
        oBufferSize = (unsigned long long)dstDescPtr->h * (unsigned long long)dstDescPtr->w * (unsigned long long)dstDescPtr->c * (unsigned long long)noOfImages;

        unsigned long long ioBufferSizeInBytes_f32 = (ioBufferSize * 4) + srcDescPtr->offsetInBytes;
        unsigned long long oBufferSizeInBytes_f32 = (oBufferSize * 4)+ dstDescPtr->offsetInBytes;

        Rpp32f *d_input,*d_output;
        hipMalloc(&d_input,ioBufferSizeInBytes_f32);
        hipMalloc(&d_output,oBufferSizeInBytes_f32);
        Rpp32f *offsetted_input, *offsetted_output;
        offsetted_input = d_input + srcDescPtr->offsetInBytes;
        Rpp32f *temp_in = (Rpp32f *)pSrc;

        Rpp32u elementsInRow = oSrcSize.width * srcDescPtr->c;

        for (int j = 0; j < oSrcSize.height; j++)
        {
            hipMemcpy(offsetted_input, temp_in, elementsInRow * sizeof (Rpp32f), hipMemcpyDeviceToDevice);
            temp_in += elementsInRow;
            offsetted_input += srcDescPtr->strides.hStride;
        }

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        status = rppt_warp_affine_gpu(d_input, srcDescPtr, d_output, dstDescPtr, affineTensor, interpolationType, roiTensorPtrSrc, roiTypeSrc, handle);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        Rpp32u elementsInRowMax = dstDescPtr->w * dstDescPtr->c;
        Rpp32u elementsInRowout = oSrcSize.width * dstDescPtr->c;
        offsetted_output = d_output + dstDescPtr->offsetInBytes;
        Rpp32f *temp_output = pDst;
        for (int k = 0; k < oSrcSize.height; k++)
        {
            hipMemcpy(temp_output, offsetted_output, elementsInRowout * sizeof (Rpp32f),hipMemcpyDeviceToDevice);
            temp_output += elementsInRowout;
            offsetted_output += elementsInRowMax;
        }

        hipFree(d_input);
        hipFree(d_output);
        hipHostFree(roiTensorPtrSrc);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiWarpAffine_32f_C1R(const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp32f *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3], int eInterpolation)
{
        int noOfImages = 1;
        int ip_channel = 1;
        RpptDesc srcDesc, dstDesc;
        RpptDescPtr srcDescPtr, dstDescPtr;
        srcDescPtr = &srcDesc;
        dstDescPtr = &dstDesc;

        srcDescPtr->layout = RpptLayout::NCHW;//pln
        dstDescPtr->layout = RpptLayout::NCHW;
        srcDescPtr->dataType = RpptDataType::F32;
        dstDescPtr->dataType = RpptDataType::F32;

        srcDescPtr->numDims = 4;
        dstDescPtr->numDims = 4;

        srcDescPtr->offsetInBytes = 64;
        dstDescPtr->offsetInBytes = 0;

        srcDescPtr->n = noOfImages;
        srcDescPtr->h = oSrcSize.height;
        srcDescPtr->w = oSrcSize.width;
        srcDescPtr->c = ip_channel;

        dstDescPtr->n = noOfImages;
        dstDescPtr->h = oSrcSize.height;
        dstDescPtr->w = oSrcSize.width;
        dstDescPtr->c = ip_channel;

    // Optionally set w stride as a multiple of 8 for src/dst

        srcDescPtr->w = ((srcDescPtr->w / 8) * 8) + 8;
        dstDescPtr->w = ((dstDescPtr->w / 8) * 8) + 8;

    // Set n/c/h/w strides for src/dst

        srcDescPtr->strides.nStride = ip_channel * srcDescPtr->w * srcDescPtr->h;
        srcDescPtr->strides.cStride = srcDescPtr->w * srcDescPtr->h;
        srcDescPtr->strides.hStride = srcDescPtr->w;
        srcDescPtr->strides.wStride = 1;

        dstDescPtr->strides.nStride = ip_channel * dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.cStride = dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.hStride = dstDescPtr->w;
        dstDescPtr->strides.wStride = 1;

        RpptROI *roiTensorPtrSrc;
        hipHostMalloc(&roiTensorPtrSrc, noOfImages * sizeof(RpptROI));

        RpptRoiType roiTypeSrc;
        roiTypeSrc = RpptRoiType::XYWH;

        roiTensorPtrSrc[0].xywhROI.xy.x = 0;
        roiTensorPtrSrc[0].xywhROI.xy.y = 0;
        roiTensorPtrSrc[0].xywhROI.roiWidth = oSrcSize.width;
        roiTensorPtrSrc[0].xywhROI.roiHeight = oSrcSize.height;

        Rpp32f6 affineTensor_f6[1];
        Rpp32f *affineTensor = (Rpp32f *)affineTensor_f6;
        affineTensor_f6[0].data[0] = aCoeffs[0][0];
        affineTensor_f6[0].data[1] = aCoeffs[0][1];
        affineTensor_f6[0].data[2] = aCoeffs[0][2];
        affineTensor_f6[0].data[3] = aCoeffs[1][0];
        affineTensor_f6[0].data[4] = aCoeffs[1][1];
        affineTensor_f6[0].data[5] = aCoeffs[1][2];

        RpptInterpolationType interpolationType;

        if(eInterpolation == NPPI_INTER_NN)
        {
            interpolationType = RpptInterpolationType::NEAREST_NEIGHBOR;
        }else if(eInterpolation == NPPI_INTER_LINEAR)
        {
            interpolationType = RpptInterpolationType::BILINEAR;
        }

        unsigned long long ioBufferSize = 0;
        unsigned long long oBufferSize = 0;

        ioBufferSize = (unsigned long long)srcDescPtr->h * (unsigned long long)srcDescPtr->w * (unsigned long long)srcDescPtr->c * (unsigned long long)noOfImages;
        oBufferSize = (unsigned long long)dstDescPtr->h * (unsigned long long)dstDescPtr->w * (unsigned long long)dstDescPtr->c * (unsigned long long)noOfImages;

        unsigned long long ioBufferSizeInBytes_f32 = (ioBufferSize * 4) + srcDescPtr->offsetInBytes;
        unsigned long long oBufferSizeInBytes_f32 = (oBufferSize * 4)+ dstDescPtr->offsetInBytes;

        Rpp32f *d_input,*d_output;
        hipMalloc(&d_input,ioBufferSizeInBytes_f32);
        hipMalloc(&d_output,oBufferSizeInBytes_f32);
        Rpp32f *offsetted_input, *offsetted_output;
        offsetted_input = d_input + srcDescPtr->offsetInBytes;
        Rpp32f *temp_in = (Rpp32f *)pSrc;

        Rpp32u elementsInRow = oSrcSize.width * srcDescPtr->c;

        for (int j = 0; j < oSrcSize.height; j++)
        {
            hipMemcpy(offsetted_input, temp_in, elementsInRow * sizeof (Rpp32f), hipMemcpyDeviceToDevice);
            temp_in += elementsInRow;
            offsetted_input += srcDescPtr->strides.hStride;
        }

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        status = rppt_warp_affine_gpu(d_input, srcDescPtr, d_output, dstDescPtr, affineTensor, interpolationType, roiTensorPtrSrc, roiTypeSrc, handle);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        Rpp32u elementsInRowMax = dstDescPtr->w * dstDescPtr->c;
        Rpp32u elementsInRowout = oSrcSize.width * dstDescPtr->c;
        offsetted_output = d_output + dstDescPtr->offsetInBytes;
        Rpp32f *temp_output = pDst;
        for (int k = 0; k < oSrcSize.height; k++)
        {
            hipMemcpy(temp_output, offsetted_output, elementsInRowout * sizeof (Rpp32f),hipMemcpyDeviceToDevice);
            temp_output += elementsInRowout;
            offsetted_output += elementsInRowMax;
        }

        hipFree(d_input);
        hipFree(d_output);
        hipHostFree(roiTensorPtrSrc);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiWarpAffine_32f_C3R_Ctx(const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp32f *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3], int eInterpolation, NppStreamContext nppStreamCtx)
{
        int noOfImages = 1;
        int ip_channel = 3;
        RpptDesc srcDesc, dstDesc;
        RpptDescPtr srcDescPtr, dstDescPtr;
        srcDescPtr = &srcDesc;
        dstDescPtr = &dstDesc;

        srcDescPtr->layout = RpptLayout::NHWC;//pkd
        dstDescPtr->layout = RpptLayout::NHWC;
        srcDescPtr->dataType = RpptDataType::F32;
        dstDescPtr->dataType = RpptDataType::F32;

        srcDescPtr->numDims = 4;
        dstDescPtr->numDims = 4;

        srcDescPtr->offsetInBytes = 64;
        dstDescPtr->offsetInBytes = 0;

        srcDescPtr->n = noOfImages;
        srcDescPtr->h = oSrcSize.height;
        srcDescPtr->w = oSrcSize.width;
        srcDescPtr->c = ip_channel;

        dstDescPtr->n = noOfImages;
        dstDescPtr->h = oSrcSize.height;
        dstDescPtr->w = oSrcSize.width;
        dstDescPtr->c = ip_channel;

    // Optionally set w stride as a multiple of 8 for src/dst

        srcDescPtr->w = ((srcDescPtr->w / 8) * 8) + 8;
        dstDescPtr->w = ((dstDescPtr->w / 8) * 8) + 8;

    // Set n/c/h/w strides for src/dst

        srcDescPtr->strides.nStride = srcDescPtr->c * srcDescPtr->w * srcDescPtr->h;
        srcDescPtr->strides.hStride = srcDescPtr->c * srcDescPtr->w;
        srcDescPtr->strides.wStride = srcDescPtr->c;
        srcDescPtr->strides.cStride = 1;

        dstDescPtr->strides.nStride = dstDescPtr->c * dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.hStride = dstDescPtr->c * dstDescPtr->w;
        dstDescPtr->strides.wStride = dstDescPtr->c;
        dstDescPtr->strides.cStride = 1;

        RpptROI *roiTensorPtrSrc;
        hipHostMalloc(&roiTensorPtrSrc, noOfImages * sizeof(RpptROI));

        RpptRoiType roiTypeSrc;
        roiTypeSrc = RpptRoiType::XYWH;

        roiTensorPtrSrc[0].xywhROI.xy.x = 0;
        roiTensorPtrSrc[0].xywhROI.xy.y = 0;
        roiTensorPtrSrc[0].xywhROI.roiWidth = oSrcSize.width;
        roiTensorPtrSrc[0].xywhROI.roiHeight = oSrcSize.height;

        Rpp32f6 affineTensor_f6[1];
        Rpp32f *affineTensor = (Rpp32f *)affineTensor_f6;
        affineTensor_f6[0].data[0] = aCoeffs[0][0];
        affineTensor_f6[0].data[1] = aCoeffs[0][1];
        affineTensor_f6[0].data[2] = aCoeffs[0][2];
        affineTensor_f6[0].data[3] = aCoeffs[1][0];
        affineTensor_f6[0].data[4] = aCoeffs[1][1];
        affineTensor_f6[0].data[5] = aCoeffs[1][2];

        RpptInterpolationType interpolationType;

        if(eInterpolation == NPPI_INTER_NN)
        {
            interpolationType = RpptInterpolationType::NEAREST_NEIGHBOR;
        }else if(eInterpolation == NPPI_INTER_LINEAR)
        {
            interpolationType = RpptInterpolationType::BILINEAR;
        }

        unsigned long long ioBufferSize = 0;
	    unsigned long long oBufferSize = 0;

        ioBufferSize = (unsigned long long)srcDescPtr->h * (unsigned long long)srcDescPtr->w * (unsigned long long)srcDescPtr->c * (unsigned long long)noOfImages;
        oBufferSize = (unsigned long long)dstDescPtr->h * (unsigned long long)dstDescPtr->w * (unsigned long long)dstDescPtr->c * (unsigned long long)noOfImages;

        unsigned long long ioBufferSizeInBytes_f32 = (ioBufferSize * 4) + srcDescPtr->offsetInBytes;
        unsigned long long oBufferSizeInBytes_f32 = (oBufferSize * 4)+ dstDescPtr->offsetInBytes;

        Rpp32f *d_input,*d_output;
        hipMalloc(&d_input,ioBufferSizeInBytes_f32);
        hipMalloc(&d_output,oBufferSizeInBytes_f32);
        Rpp32f *offsetted_input, *offsetted_output;
        offsetted_input = d_input + srcDescPtr->offsetInBytes;
        Rpp32f *temp_in = (Rpp32f *)pSrc;

        Rpp32u elementsInRow = oSrcSize.width * srcDescPtr->c;

        for (int j = 0; j < oSrcSize.height; j++)
        {
            hipMemcpy(offsetted_input, temp_in, elementsInRow * sizeof (Rpp32f), hipMemcpyDeviceToDevice);
            temp_in += elementsInRow;
            offsetted_input += srcDescPtr->strides.hStride;
        }

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        status = rppt_warp_affine_gpu(d_input, srcDescPtr, d_output, dstDescPtr, affineTensor, interpolationType, roiTensorPtrSrc, roiTypeSrc, handle);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        Rpp32u elementsInRowMax = dstDescPtr->w * dstDescPtr->c;
        Rpp32u elementsInRowout = oSrcSize.width * dstDescPtr->c;
        offsetted_output = d_output + dstDescPtr->offsetInBytes;
        Rpp32f *temp_output = pDst;
        for (int k = 0; k < oSrcSize.height; k++)
        {
            hipMemcpy(temp_output, offsetted_output, elementsInRowout * sizeof (Rpp32f),hipMemcpyDeviceToDevice);
            temp_output += elementsInRowout;
            offsetted_output += elementsInRowMax;
        }

        hipFree(d_input);
        hipFree(d_output);
        hipHostFree(roiTensorPtrSrc);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiWarpAffine_32f_C3R(const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp32f *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3], int eInterpolation)
{
        int noOfImages = 1;
        int ip_channel = 3;
        RpptDesc srcDesc, dstDesc;
        RpptDescPtr srcDescPtr, dstDescPtr;
        srcDescPtr = &srcDesc;
        dstDescPtr = &dstDesc;

        srcDescPtr->layout = RpptLayout::NHWC;//pkd
        dstDescPtr->layout = RpptLayout::NHWC;
        srcDescPtr->dataType = RpptDataType::F32;
        dstDescPtr->dataType = RpptDataType::F32;

        srcDescPtr->numDims = 4;
        dstDescPtr->numDims = 4;

        srcDescPtr->offsetInBytes = 64;
        dstDescPtr->offsetInBytes = 0;

        srcDescPtr->n = noOfImages;
        srcDescPtr->h = oSrcSize.height;
        srcDescPtr->w = oSrcSize.width;
        srcDescPtr->c = ip_channel;

        dstDescPtr->n = noOfImages;
        dstDescPtr->h = oSrcSize.height;
        dstDescPtr->w = oSrcSize.width;
        dstDescPtr->c = ip_channel;

    // Optionally set w stride as a multiple of 8 for src/dst

        srcDescPtr->w = ((srcDescPtr->w / 8) * 8) + 8;
        dstDescPtr->w = ((dstDescPtr->w / 8) * 8) + 8;

    // Set n/c/h/w strides for src/dst

        srcDescPtr->strides.nStride = srcDescPtr->c * srcDescPtr->w * srcDescPtr->h;
        srcDescPtr->strides.hStride = srcDescPtr->c * srcDescPtr->w;
        srcDescPtr->strides.wStride = srcDescPtr->c;
        srcDescPtr->strides.cStride = 1;

        dstDescPtr->strides.nStride = dstDescPtr->c * dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.hStride = dstDescPtr->c * dstDescPtr->w;
        dstDescPtr->strides.wStride = dstDescPtr->c;
        dstDescPtr->strides.cStride = 1;

        RpptROI *roiTensorPtrSrc;
        hipHostMalloc(&roiTensorPtrSrc, noOfImages * sizeof(RpptROI));

        RpptRoiType roiTypeSrc;
        roiTypeSrc = RpptRoiType::XYWH;

        roiTensorPtrSrc[0].xywhROI.xy.x = 0;
        roiTensorPtrSrc[0].xywhROI.xy.y = 0;
        roiTensorPtrSrc[0].xywhROI.roiWidth = oSrcSize.width;
        roiTensorPtrSrc[0].xywhROI.roiHeight = oSrcSize.height;

        Rpp32f6 affineTensor_f6[1];
        Rpp32f *affineTensor = (Rpp32f *)affineTensor_f6;
        affineTensor_f6[0].data[0] = aCoeffs[0][0];
        affineTensor_f6[0].data[1] = aCoeffs[0][1];
        affineTensor_f6[0].data[2] = aCoeffs[0][2];
        affineTensor_f6[0].data[3] = aCoeffs[1][0];
        affineTensor_f6[0].data[4] = aCoeffs[1][1];
        affineTensor_f6[0].data[5] = aCoeffs[1][2];

        RpptInterpolationType interpolationType;

        if(eInterpolation == NPPI_INTER_NN)
        {
            interpolationType = RpptInterpolationType::NEAREST_NEIGHBOR;
        }else if(eInterpolation == NPPI_INTER_LINEAR)
        {
            interpolationType = RpptInterpolationType::BILINEAR;
        }

        unsigned long long ioBufferSize = 0;
	    unsigned long long oBufferSize = 0;

        ioBufferSize = (unsigned long long)srcDescPtr->h * (unsigned long long)srcDescPtr->w * (unsigned long long)srcDescPtr->c * (unsigned long long)noOfImages;
        oBufferSize = (unsigned long long)dstDescPtr->h * (unsigned long long)dstDescPtr->w * (unsigned long long)dstDescPtr->c * (unsigned long long)noOfImages;

        unsigned long long ioBufferSizeInBytes_f32 = (ioBufferSize * 4) + srcDescPtr->offsetInBytes;
        unsigned long long oBufferSizeInBytes_f32 = (oBufferSize * 4)+ dstDescPtr->offsetInBytes;

        Rpp32f *d_input,*d_output;
        hipMalloc(&d_input,ioBufferSizeInBytes_f32);
        hipMalloc(&d_output,oBufferSizeInBytes_f32);
        Rpp32f *offsetted_input, *offsetted_output;
        offsetted_input = d_input + srcDescPtr->offsetInBytes;
        Rpp32f *temp_in = (Rpp32f *)pSrc;

        Rpp32u elementsInRow = oSrcSize.width * srcDescPtr->c;

        for (int j = 0; j < oSrcSize.height; j++)
        {
            hipMemcpy(offsetted_input, temp_in, elementsInRow * sizeof (Rpp32f), hipMemcpyDeviceToDevice);
            temp_in += elementsInRow;
            offsetted_input += srcDescPtr->strides.hStride;
        }

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        status = rppt_warp_affine_gpu(d_input, srcDescPtr, d_output, dstDescPtr, affineTensor, interpolationType, roiTensorPtrSrc, roiTypeSrc, handle);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        Rpp32u elementsInRowMax = dstDescPtr->w * dstDescPtr->c;
        Rpp32u elementsInRowout = oSrcSize.width * dstDescPtr->c;
        offsetted_output = d_output + dstDescPtr->offsetInBytes;
        Rpp32f *temp_output = pDst;
        for (int k = 0; k < oSrcSize.height; k++)
        {
            hipMemcpy(temp_output, offsetted_output, elementsInRowout * sizeof (Rpp32f),hipMemcpyDeviceToDevice);
            temp_output += elementsInRowout;
            offsetted_output += elementsInRowMax;
        }

        hipFree(d_input);
        hipFree(d_output);
        hipHostFree(roiTensorPtrSrc);

        return(hipRppStatusTocudaNppStatus(status));
}

//NppStatus nppiWarpAffine_32f_P3R_Ctx(const Npp32f *pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp32f *pDst[3], int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3], int eInterpolation, NppStreamContext nppStreamCtx)


//NppStatus nppiWarpAffine_32f_P3R(const Npp32f *pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp32f *pDst[3], int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3], int eInterpolation)

/*NppStatus nppiWarpAffine_8u_P3R_Ctx(const Npp8u *pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp8u *pDst[3], int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3], int eInterpolation, NppStreamContext nppStreamCtx)
{
        //pln3
        int noOfImages = 1;
        int ip_channel = 3;
        RpptDesc srcDesc, dstDesc;
        RpptDescPtr srcDescPtr, dstDescPtr;
        srcDescPtr = &srcDesc;
        dstDescPtr = &dstDesc;

        srcDescPtr->layout = RpptLayout::NCHW;
        dstDescPtr->layout = RpptLayout::NCHW;
        srcDescPtr->dataType = RpptDataType::U8;
        dstDescPtr->dataType = RpptDataType::U8;

        srcDescPtr->numDims = 4;
        dstDescPtr->numDims = 4;

        srcDescPtr->offsetInBytes = 64;
        dstDescPtr->offsetInBytes = 0;

        srcDescPtr->n = noOfImages;
        srcDescPtr->h = oSrcSize.height;
        srcDescPtr->w = oSrcSize.width;
        srcDescPtr->c = ip_channel;

        dstDescPtr->n = noOfImages;
        dstDescPtr->h = oSrcSize.height;
        dstDescPtr->w = oSrcSize.width;
        dstDescPtr->c = ip_channel;

    // Optionally set w stride as a multiple of 8 for src/dst

        srcDescPtr->w = ((srcDescPtr->w / 8) * 8) + 8;
        dstDescPtr->w = ((dstDescPtr->w / 8) * 8) + 8;

    // Set n/c/h/w strides for src/dst
        RpptROI *roiTensorPtrSrc;
        hipHostMalloc(&roiTensorPtrSrc, noOfImages * sizeof(RpptROI));
        roiTensorPtrSrc[0].xywhROI.xy.x = 0;
        roiTensorPtrSrc[0].xywhROI.xy.y = 0;
        roiTensorPtrSrc[0].xywhROI.roiWidth = oSrcSize.width;
        roiTensorPtrSrc[0].xywhROI.roiHeight = oSrcSize.height;
        RpptRoiType roiTypeSrc;
        roiTypeSrc = RpptRoiType::XYWH;

        srcDescPtr->strides.nStride = srcDescPtr->c * srcDescPtr->w * srcDescPtr->h;
	srcDescPtr->strides.cStride = srcDescPtr->w * srcDescPtr->h;
	srcDescPtr->strides.hStride = srcDescPtr->w;
	srcDescPtr->strides.wStride = 1;

        dstDescPtr->strides.nStride = dstDescPtr->c * dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.cStride = dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.hStride = dstDescPtr->w;
        dstDescPtr->strides.wStride = 1;
		
	RppStatus status;
        Rpp32f6 affineTensor_f6[1];
        Rpp32f *affineTensor = (Rpp32f *)affineTensor_f6;
        affineTensor_f6[0].data[0] = aCoeffs[0][0];
        affineTensor_f6[0].data[1] = aCoeffs[0][1];
        affineTensor_f6[0].data[2] = aCoeffs[0][2];
        affineTensor_f6[0].data[3] = aCoeffs[1][0];
        affineTensor_f6[0].data[4] = aCoeffs[1][1];
        affineTensor_f6[0].data[5] = aCoeffs[1][2];

        RpptInterpolationType interpolationType;

        if(eInterpolation == NPPI_INTER_NN)
        {
                interpolationType = RpptInterpolationType::NEAREST_NEIGHBOR;
        }else if(eInterpolation == NPPI_INTER_LINEAR)
        {
                interpolationType = RpptInterpolationType::BILINEAR;
        }

        unsigned long long ioBufferSize = 0;
        unsigned long long oBufferSize = 0;

        ioBufferSize = (unsigned long long)srcDescPtr->h * (unsigned long long)srcDescPtr->w * (unsigned long long)srcDescPtr->c * (unsigned long long)noOfImages;
        oBufferSize = (unsigned long long)dstDescPtr->h * (unsigned long long)dstDescPtr->w * (unsigned long long)dstDescPtr->c * (unsigned long long)noOfImages;

        unsigned long long ioBufferSizeInBytes_u8 = ioBufferSize + srcDescPtr->offsetInBytes;
        unsigned long long oBufferSizeInBytes_u8 = oBufferSize + dstDescPtr->offsetInBytes;

        Rpp8u *d_input,*d_output;
        hipMalloc(&d_input,ioBufferSizeInBytes_u8);
        hipMalloc(&d_output,oBufferSizeInBytes_u8);
	Rpp8u *offsetted_input, *offsetted_output;
        offsetted_input = d_input + srcDescPtr->offsetInBytes;
        
	Rpp8u *temp_in;
        hipMalloc(&temp_in, oSrcSize.height * oSrcSize.width * ip_channel);
        for (int i = 0; i < ip_channel; i++)
        {
            hipMemcpy(temp_in + (i * oSrcSize.height * oSrcSize.width), pSrc[i], oSrcSize.height * oSrcSize.width, hipMemcpyDeviceToDevice);
        }

        Rpp32u elementsInRow = oSrcSize.width * srcDescPtr->c;

        for (int j = 0; j < oSrcSize.height; j++)
        {
            hipMemcpy(offsetted_input, temp_in, elementsInRow * sizeof (Rpp8u), hipMemcpyDeviceToDevice);
            temp_in += elementsInRow;
            offsetted_input += srcDescPtr->w * srcDescPtr->c;
        }

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);
        clock_t start, end;
        double gpu_time_used;

        start = clock();
        status = rppt_warp_affine_gpu(d_input, srcDescPtr, d_output, dstDescPtr, affineTensor, interpolationType, roiTensorPtrSrc, roiTypeSrc, handle);
        hipDeviceSynchronize();
        end = clock();

        printf("\nnppiWarpAffine_8u_P3R_Ctx is %d\n", status);
        gpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
        std::cout << "\nGPU Time - WarpAffine_8u : " << gpu_time_used;
        printf("\n");

        rppDestroyGPU(handle);
        Rpp32u elementsInRowMax = dstDescPtr->w * dstDescPtr->c;
        Rpp32u elementsInRowout = oSrcSize.width * dstDescPtr->c;
        offsetted_output = d_output + dstDescPtr->offsetInBytes;
        Rpp8u *temp_output;
        hipMalloc(&temp_output, oSrcSize.height * oSrcSize.width * ip_channel);
        for (int k = 0; k < oSrcSize.height; k++)
        {
            hipMemcpy(temp_output + (k * elementsInRowout), offsetted_output, elementsInRowout * sizeof (Rpp8u),hipMemcpyDeviceToDevice);
            //temp_output += elementsInRowout;
            offsetted_output += elementsInRowMax;
        }
        for (int m = 0; m < ip_channel; m++)
        {
            hipMemcpy(pDst[m],temp_output + (m * oSrcSize.height * oSrcSize.width), oSrcSize.height * oSrcSize.width, hipMemcpyDeviceToDevice);
        }


        hipFree(d_input);
        hipFree(d_output);
        hipHostFree(roiTensorPtrSrc);
	hipFree(temp_in);
        hipFree(temp_output);

        return(hipRppStatusTocudaNppStatus(status));
}*/

/*NppStatus nppiWarpAffine_8u_P3R(const Npp8u *pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp8u *pDst[3], int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3], int eInterpolation)
{
        //pln3
        int noOfImages = 1;
        int ip_channel = 3;
        RpptDesc srcDesc, dstDesc;
        RpptDescPtr srcDescPtr, dstDescPtr;
        srcDescPtr = &srcDesc;
        dstDescPtr = &dstDesc;

        srcDescPtr->layout = RpptLayout::NCHW;
        dstDescPtr->layout = RpptLayout::NCHW;
        srcDescPtr->dataType = RpptDataType::U8;
        dstDescPtr->dataType = RpptDataType::U8;

        srcDescPtr->numDims = 4;
        dstDescPtr->numDims = 4;

        srcDescPtr->offsetInBytes = 64;
        dstDescPtr->offsetInBytes = 0;

        srcDescPtr->n = noOfImages;
        srcDescPtr->h = oSrcSize.height;
        srcDescPtr->w = oSrcSize.width;
        srcDescPtr->c = ip_channel;

        dstDescPtr->n = noOfImages;
        dstDescPtr->h = oSrcSize.height;
        dstDescPtr->w = oSrcSize.width;
        dstDescPtr->c = ip_channel;

    // Optionally set w stride as a multiple of 8 for src/dst

        srcDescPtr->w = ((srcDescPtr->w / 8) * 8) + 8;
        dstDescPtr->w = ((dstDescPtr->w / 8) * 8) + 8;

    // Set n/c/h/w strides for src/dst
        RpptROI *roiTensorPtrSrc;
        hipHostMalloc(&roiTensorPtrSrc, noOfImages * sizeof(RpptROI));
        roiTensorPtrSrc[0].xywhROI.xy.x = 0;
        roiTensorPtrSrc[0].xywhROI.xy.y = 0;
        roiTensorPtrSrc[0].xywhROI.roiWidth = oSrcSize.width;
        roiTensorPtrSrc[0].xywhROI.roiHeight = oSrcSize.height;
        RpptRoiType roiTypeSrc;
        roiTypeSrc = RpptRoiType::XYWH;

        srcDescPtr->strides.nStride = srcDescPtr->c * srcDescPtr->w * srcDescPtr->h;
	srcDescPtr->strides.cStride = srcDescPtr->w * srcDescPtr->h;
	srcDescPtr->strides.hStride = srcDescPtr->w;
	srcDescPtr->strides.wStride = 1;

        dstDescPtr->strides.nStride = dstDescPtr->c * dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.cStride = dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.hStride = dstDescPtr->w;
        dstDescPtr->strides.wStride = 1;
		
	RppStatus status;
        Rpp32f6 affineTensor_f6[1];
        Rpp32f *affineTensor = (Rpp32f *)affineTensor_f6;
        affineTensor_f6[0].data[0] = aCoeffs[0][0];
        affineTensor_f6[0].data[1] = aCoeffs[0][1];
        affineTensor_f6[0].data[2] = aCoeffs[0][2];
        affineTensor_f6[0].data[3] = aCoeffs[1][0];
        affineTensor_f6[0].data[4] = aCoeffs[1][1];
        affineTensor_f6[0].data[5] = aCoeffs[1][2];

        RpptInterpolationType interpolationType;

        if(eInterpolation == NPPI_INTER_NN)
        {
                interpolationType = RpptInterpolationType::NEAREST_NEIGHBOR;
        }else if(eInterpolation == NPPI_INTER_LINEAR)
        {
                interpolationType = RpptInterpolationType::BILINEAR;
        }

        unsigned long long ioBufferSize = 0;
        unsigned long long oBufferSize = 0;

        ioBufferSize = (unsigned long long)srcDescPtr->h * (unsigned long long)srcDescPtr->w * (unsigned long long)srcDescPtr->c * (unsigned long long)noOfImages;
        oBufferSize = (unsigned long long)dstDescPtr->h * (unsigned long long)dstDescPtr->w * (unsigned long long)dstDescPtr->c * (unsigned long long)noOfImages;

        unsigned long long ioBufferSizeInBytes_u8 = ioBufferSize + srcDescPtr->offsetInBytes;
        unsigned long long oBufferSizeInBytes_u8 = oBufferSize + dstDescPtr->offsetInBytes;

        Rpp8u *d_input,*d_output;
        hipMalloc(&d_input,ioBufferSizeInBytes_u8);
        hipMalloc(&d_output,oBufferSizeInBytes_u8);
	Rpp8u *offsetted_input, *offsetted_output;
        offsetted_input = d_input + srcDescPtr->offsetInBytes;
        
	Rpp8u *temp_in;
        hipMalloc(&temp_in, oSrcSize.height * oSrcSize.width * ip_channel);
        for (int i = 0; i < ip_channel; i++)
        {
            hipMemcpy(temp_in + (i * oSrcSize.height * oSrcSize.width), pSrc[i], oSrcSize.height * oSrcSize.width, hipMemcpyDeviceToDevice);
        }

        Rpp32u elementsInRow = oSrcSize.width * srcDescPtr->c;

        for (int j = 0; j < oSrcSize.height; j++)
        {
            hipMemcpy(offsetted_input, temp_in, elementsInRow * sizeof (Rpp8u), hipMemcpyDeviceToDevice);
            temp_in += elementsInRow;
            offsetted_input += srcDescPtr->w * srcDescPtr->c;
        }

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);
        clock_t start, end;
        double gpu_time_used;

        start = clock();
        status = rppt_warp_affine_gpu(d_input, srcDescPtr, d_output, dstDescPtr, affineTensor, interpolationType, roiTensorPtrSrc, roiTypeSrc, handle);
        hipDeviceSynchronize();
        end = clock();

        printf("\nnppiWarpAffine_8u_P3R is %d\n", status);
        gpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
        std::cout << "\nGPU Time - WarpAffine_8u : " << gpu_time_used;
        printf("\n");

        rppDestroyGPU(handle);
        Rpp32u elementsInRowMax = dstDescPtr->w * dstDescPtr->c;
        Rpp32u elementsInRowout = oSrcSize.width * dstDescPtr->c;
        offsetted_output = d_output + dstDescPtr->offsetInBytes;
        Rpp8u *temp_output;
        hipMalloc(&temp_output, oSrcSize.height * oSrcSize.width * ip_channel);
        for (int k = 0; k < oSrcSize.height; k++)
        {
            hipMemcpy(temp_output + (k * elementsInRowout), offsetted_output, elementsInRowout * sizeof (Rpp8u),hipMemcpyDeviceToDevice);
            //temp_output += elementsInRowout;
            offsetted_output += elementsInRowMax;
        }
        for (int m = 0; m < ip_channel; m++)
        {
            hipMemcpy(pDst[m],temp_output + (m * oSrcSize.height * oSrcSize.width), oSrcSize.height * oSrcSize.width, hipMemcpyDeviceToDevice);
        }


        hipFree(d_input);
        hipFree(d_output);
        hipHostFree(roiTensorPtrSrc);
	hipFree(temp_in);
        hipFree(temp_output);

        return(hipRppStatusTocudaNppStatus(status));
}*/

/******************** flip ********************/

RppStatus rppt_flip_gpu(RppPtr_t srcPtr,
                        RpptDescPtr srcDescPtr,
                        RppPtr_t dstPtr,
                        RpptDescPtr dstDescPtr,
                        Rpp32u *horizontalTensor,
                        Rpp32u *verticalTensor,
                        RpptROIPtr roiTensorPtrSrc,
                        RpptRoiType roiType,
                        rppHandle_t rppHandle)
{
#ifdef HIP_COMPILE
    Rpp32u paramIndex = 0;
    copy_param_uint(horizontalTensor, rpp::deref(rppHandle), paramIndex++);
    copy_param_uint(verticalTensor, rpp::deref(rppHandle), paramIndex++);

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        hip_exec_flip_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                             srcDescPtr,
                             static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                             dstDescPtr,
                             roiTensorPtrSrc,
                             roiType,
                             rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        hip_exec_flip_tensor((half*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                             srcDescPtr,
                             (half*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                             dstDescPtr,
                             roiTensorPtrSrc,
                             roiType,
                             rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        hip_exec_flip_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                             srcDescPtr,
                             (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                             dstDescPtr,
                             roiTensorPtrSrc,
                             roiType,
                             rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        hip_exec_flip_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                             srcDescPtr,
                             static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                             dstDescPtr,
                             roiTensorPtrSrc,
                             roiType,
                             rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
#elif defined(OCL_COMPILE)
    return RPP_ERROR_NOT_IMPLEMENTED;
#endif // backend
}
/*
NppStatus nppiMirror_8u_C3R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oROI, NppiAxis flip)
{
        int noOfImages = 1;
        int ip_channel = 3;
        RpptDesc srcDesc, dstDesc;
        RpptDescPtr srcDescPtr, dstDescPtr;
        srcDescPtr = &srcDesc;
        dstDescPtr = &dstDesc;

        srcDescPtr->layout = RpptLayout::NHWC;//pkd
        dstDescPtr->layout = RpptLayout::NHWC;
        srcDescPtr->dataType = RpptDataType::U8;
        dstDescPtr->dataType = RpptDataType::U8;

        srcDescPtr->numDims = 4;
        dstDescPtr->numDims = 4;

        srcDescPtr->offsetInBytes = 64;
        dstDescPtr->offsetInBytes = 0;

        srcDescPtr->n = noOfImages;
        srcDescPtr->h = oROI.height;
        srcDescPtr->w = oROI.width;
        srcDescPtr->c = ip_channel;

        dstDescPtr->n = noOfImages;
        dstDescPtr->h = oROI.height;
        dstDescPtr->w = oROI.width;
        dstDescPtr->c = ip_channel;

    // Optionally set w stride as a multiple of 8 for src/dst

        srcDescPtr->w = ((srcDescPtr->w / 8) * 8) + 8;
        dstDescPtr->w = ((dstDescPtr->w / 8) * 8) + 8;

    // Set n/c/h/w strides for src/dst
        RpptROI *roiTensorPtrSrc;
        hipHostMalloc(&roiTensorPtrSrc, noOfImages * sizeof(RpptROI));

        RpptRoiType roiTypeSrc;
        roiTypeSrc = RpptRoiType::XYWH;

        roiTensorPtrSrc[0].xywhROI.xy.x = 0;
        roiTensorPtrSrc[0].xywhROI.xy.y = 0;
        roiTensorPtrSrc[0].xywhROI.roiWidth = oROI.width;
        roiTensorPtrSrc[0].xywhROI.roiHeight = oROI.height;

        srcDescPtr->strides.nStride = srcDescPtr->c * srcDescPtr->w * srcDescPtr->h;
        srcDescPtr->strides.hStride = srcDescPtr->c * srcDescPtr->w;
        srcDescPtr->strides.wStride = srcDescPtr->c;
        srcDescPtr->strides.cStride = 1;

        dstDescPtr->strides.nStride = dstDescPtr->c * dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.hStride = dstDescPtr->c * dstDescPtr->w;
        dstDescPtr->strides.wStride = dstDescPtr->c;
        dstDescPtr->strides.cStride = 1;
		
        unsigned long long ioBufferSize = 0;
        unsigned long long oBufferSize = 0;

        ioBufferSize = (unsigned long long)srcDescPtr->h * (unsigned long long)srcDescPtr->w * (unsigned long long)srcDescPtr->c * (unsigned long long)noOfImages;
        oBufferSize = (unsigned long long)dstDescPtr->h * (unsigned long long)dstDescPtr->w * (unsigned long long)dstDescPtr->c * (unsigned long long)noOfImages;

        unsigned long long ioBufferSizeInBytes_u8 = ioBufferSize + srcDescPtr->offsetInBytes;
        unsigned long long oBufferSizeInBytes_u8 = oBufferSize + dstDescPtr->offsetInBytes;
		
	Rpp32u horizontalFlag[1] = {0};
	Rpp32u	verticalFlag[1] = {0};
	if(flip == NPP_HORIZONTAL_AXIS)
	{
		horizontalFlag[0] = 1;
	}else if(flip == NPP_VERTICAL_AXIS)
	{
		verticalFlag[0] = 1;
	}else if(flip == NPP_BOTH_AXIS)
	{
		horizontalFlag[0] = 1;
		verticalFlag[0] = 1;
	}

        Rpp8u *d_input,*d_output;
	hipMalloc(&d_input,ioBufferSizeInBytes_u8);
        hipMalloc(&d_output,oBufferSizeInBytes_u8);
        Rpp8u *offsetted_input, *offsetted_output;
        offsetted_input = d_input + srcDescPtr->offsetInBytes;
        Rpp8u *temp_in = (Rpp8u *)pSrc;

        Rpp32u elementsInRow = oROI.width * srcDescPtr->c;

        for (int j = 0; j < oROI.height; j++)
        {
            hipMemcpy(offsetted_input, temp_in, elementsInRow * sizeof (Rpp8u), hipMemcpyDeviceToDevice);
            temp_in += elementsInRow;
            offsetted_input += srcDescPtr->strides.hStride;
        }
		
        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);
        clock_t start, end;
        double gpu_time_used;

        RppStatus status;
        start = clock();
        status = rppt_flip_gpu(d_input, srcDescPtr, d_output, dstDescPtr, horizontalFlag, verticalFlag, roiTensorPtrSrc, roiTypeSrc, handle);
        hipDeviceSynchronize();
        end = clock();

        printf("\nnppiMirror_8u_C3R is %d\n", status);
        gpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
        std::cout << "\nGPU Time - Mirror_8u : " << gpu_time_used;
        printf("\n");
      
        rppDestroyGPU(handle);
        Rpp32u elementsInRowMax = dstDescPtr->w * dstDescPtr->c;
        Rpp32u elementsInRowout = oROI.width * dstDescPtr->c;
        offsetted_output = d_output + dstDescPtr->offsetInBytes;
        Rpp8u *temp_output = pDst;
        for (int k = 0; k < oROI.height; k++)
        {
            hipMemcpy(temp_output, offsetted_output, elementsInRowout * sizeof (Rpp8u),hipMemcpyDeviceToDevice);
            temp_output += elementsInRowout;
            offsetted_output += elementsInRowMax;
	}

	hipFree(d_input);
        hipFree(d_output);
        hipHostFree(roiTensorPtrSrc);

        return(hipRppStatusTocudaNppStatus(status));
}
*/
/******************** resize_mirror_normalize ********************/

RppStatus rppt_resize_mirror_normalize_gpu(RppPtr_t srcPtr,
                                           RpptDescPtr srcDescPtr,
                                           RppPtr_t dstPtr,
                                           RpptDescPtr dstDescPtr,
                                           RpptImagePatchPtr dstImgSizes,
                                           RpptInterpolationType interpolationType,
                                           Rpp32f *meanTensor,
                                           Rpp32f *stdDevTensor,
                                           Rpp32u *mirrorTensor,
                                           RpptROIPtr roiTensorPtrSrc,
                                           RpptRoiType roiType,
                                           rppHandle_t rppHandle)
{
#ifdef HIP_COMPILE
    if (interpolationType != RpptInterpolationType::BILINEAR)
        return RPP_ERROR_NOT_IMPLEMENTED;

    Rpp32u paramIndex = 0;
    if(srcDescPtr->c == 3)
    {
        copy_param_float3(meanTensor, rpp::deref(rppHandle), paramIndex++);
        copy_param_float3(stdDevTensor, rpp::deref(rppHandle), paramIndex++);
    }
    else if(srcDescPtr->c == 1)
    {
        copy_param_float(meanTensor, rpp::deref(rppHandle), paramIndex++);
        copy_param_float(stdDevTensor, rpp::deref(rppHandle), paramIndex++);
    }
    copy_param_uint(mirrorTensor, rpp::deref(rppHandle), paramIndex++);

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        hip_exec_resize_mirror_normalize_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                                srcDescPtr,
                                                static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                                dstDescPtr,
                                                dstImgSizes,
                                                interpolationType,
                                                roiTensorPtrSrc,
                                                roiType,
                                                rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        hip_exec_resize_mirror_normalize_tensor((half*)(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                                srcDescPtr,
                                                (half*)(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                                dstDescPtr,
                                                dstImgSizes,
                                                interpolationType,
                                                roiTensorPtrSrc,
                                                roiType,
                                                rpp::deref(rppHandle));
    }

    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        hip_exec_resize_mirror_normalize_tensor((Rpp32f*)(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                                srcDescPtr,
                                                (Rpp32f*)(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                                dstDescPtr,
                                                dstImgSizes,
                                                interpolationType,
                                                roiTensorPtrSrc,
                                                roiType,
                                                rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        hip_exec_resize_mirror_normalize_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                                srcDescPtr,
                                                static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                                dstDescPtr,
                                                dstImgSizes,
                                                interpolationType,
                                                roiTensorPtrSrc,
                                                roiType,
                                                rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        hip_exec_resize_mirror_normalize_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                                srcDescPtr,
                                                (Rpp32f *)(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                                dstDescPtr,
                                                dstImgSizes,
                                                interpolationType,
                                                roiTensorPtrSrc,
                                                roiType,
                                                rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        hip_exec_resize_mirror_normalize_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                                srcDescPtr,
                                                (half *)(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                                dstDescPtr,
                                                dstImgSizes,
                                                interpolationType,
                                                roiTensorPtrSrc,
                                                roiType,
                                                rpp::deref(rppHandle));
    }

return RPP_SUCCESS;
#elif defined(OCL_COMPILE)
    return RPP_ERROR_NOT_IMPLEMENTED;
#endif // backend
}

/******************** resize ********************/

RppStatus rppt_resize_gpu(RppPtr_t srcPtr,
                          RpptDescPtr srcDescPtr,
                          RppPtr_t dstPtr,
                          RpptDescPtr dstDescPtr,
                          RpptImagePatchPtr dstImgSizes,
                          RpptInterpolationType interpolationType,
                          RpptROIPtr roiTensorPtrSrc,
                          RpptRoiType roiType,
                          rppHandle_t rppHandle)
{
#ifdef HIP_COMPILE
    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        hip_exec_resize_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                               srcDescPtr,
                               static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                               dstDescPtr,
                               dstImgSizes,
                               interpolationType,
                               roiTensorPtrSrc,
                               roiType,
                               rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        hip_exec_resize_tensor((half*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                               srcDescPtr,
                               (half*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                               dstDescPtr,
                               dstImgSizes,
                               interpolationType,
                               roiTensorPtrSrc,
                               roiType,
                               rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        hip_exec_resize_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                               srcDescPtr,
                               (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                               dstDescPtr,
                               dstImgSizes,
                               interpolationType,
                               roiTensorPtrSrc,
                               roiType,
                               rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        hip_exec_resize_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                               srcDescPtr,
                               static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                               dstDescPtr,
                               dstImgSizes,
                               interpolationType,
                               roiTensorPtrSrc,
                               roiType,
                               rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
#elif defined(OCL_COMPILE)
    return RPP_ERROR_NOT_IMPLEMENTED;
#endif // backend
}

NppStatus nppiResize_8u_C3R(const Npp8u *pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI, 
			Npp8u *pDst, int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI, int eInterpolation)

{
        //pkd
	    int noOfImages = 1;
        int ip_channel = 3;
        RpptDesc srcDesc, dstDesc;
        RpptDescPtr srcDescPtr, dstDescPtr;
        srcDescPtr = &srcDesc;
        dstDescPtr = &dstDesc;

        srcDescPtr->layout = RpptLayout::NHWC;
        dstDescPtr->layout = RpptLayout::NHWC;
        srcDescPtr->dataType = RpptDataType::U8;
        dstDescPtr->dataType = RpptDataType::U8;

        srcDescPtr->numDims = 4;
        dstDescPtr->numDims = 4;

        srcDescPtr->offsetInBytes = 64;
        dstDescPtr->offsetInBytes = 0;

        srcDescPtr->n = noOfImages;
        srcDescPtr->h = oSrcSize.height;
        srcDescPtr->w = oSrcSize.width;
        srcDescPtr->c = ip_channel;

        dstDescPtr->n = noOfImages;
        dstDescPtr->h = oSrcSize.height;
        dstDescPtr->w = oSrcSize.width;
        dstDescPtr->c = ip_channel;

        RppStatus status;
        RpptInterpolationType interpolationType = RpptInterpolationType::BILINEAR;
		
	    if(eInterpolation == NPPI_INTER_UNDEFINED) {
		    interpolationType = RpptInterpolationType::NEAREST_NEIGHBOR; 
	    } else if (eInterpolation == NPPI_INTER_LINEAR) {
		    interpolationType = RpptInterpolationType::BILINEAR;
	    } else if (eInterpolation == NPPI_INTER_CUBIC) {
		    interpolationType = RpptInterpolationType::BICUBIC;
	    } else if(eInterpolation == NPPI_INTER_LANCZOS) {
		    interpolationType = RpptInterpolationType::LANCZOS;
	    } else {
		    status = RPP_ERROR_INVALID_ARGUMENTS;
		return(hipRppStatusTocudaNppStatus(status));
	    }

    // Optionally set w stride as a multiple of 8 for src/dst

        srcDescPtr->w = ((srcDescPtr->w / 8) * 8) + 8;
        dstDescPtr->w = ((dstDescPtr->w / 8) * 8) + 8;

    // Set n/c/h/w strides for src/dst
    	RpptROI *roiTensorPtrSrc;
        hipHostMalloc(&roiTensorPtrSrc, noOfImages * sizeof(RpptROI));
        roiTensorPtrSrc[0].xywhROI.xy.x = 0;
        roiTensorPtrSrc[0].xywhROI.xy.y = 0;
        roiTensorPtrSrc[0].xywhROI.roiWidth = oSrcSize.width;
        roiTensorPtrSrc[0].xywhROI.roiHeight = oSrcSize.height;
        RpptRoiType roiTypeSrc;
        roiTypeSrc = RpptRoiType::XYWH;
        RpptImagePatch *dstImgSizes;
        hipHostMalloc(&dstImgSizes, noOfImages * sizeof(RpptImagePatch));
        dstImgSizes[0].width = oDstSize.width;
        dstImgSizes[0].height = oDstSize.height;

        srcDescPtr->strides.nStride = srcDescPtr->c * srcDescPtr->w * srcDescPtr->h;
        srcDescPtr->strides.hStride = srcDescPtr->c * srcDescPtr->w;
        srcDescPtr->strides.wStride = srcDescPtr->c;
        srcDescPtr->strides.cStride = 1;

        dstDescPtr->strides.nStride = dstDescPtr->c * dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.hStride = dstDescPtr->c * dstDescPtr->w;
        dstDescPtr->strides.wStride = dstDescPtr->c;
        dstDescPtr->strides.cStride = 1;

        unsigned long long ioBufferSize = 0;
        unsigned long long oBufferSize = 0;

        ioBufferSize = (unsigned long long)srcDescPtr->h * (unsigned long long)srcDescPtr->w * (unsigned long long)srcDescPtr->c * (unsigned long long)noOfImages;
        oBufferSize = (unsigned long long)dstDescPtr->h * (unsigned long long)dstDescPtr->w * (unsigned long long)dstDescPtr->c * (unsigned long long)noOfImages;

        unsigned long long ioBufferSizeInBytes_u8 = ioBufferSize + srcDescPtr->offsetInBytes;
        unsigned long long oBufferSizeInBytes_u8 = oBufferSize + dstDescPtr->offsetInBytes;

        Rpp8u *d_input,*d_output;
	    hipMalloc(&d_input,ioBufferSizeInBytes_u8);
        hipMalloc(&d_output,oBufferSizeInBytes_u8);
        Rpp8u *offsetted_input, *offsetted_output;
        offsetted_input = d_input + srcDescPtr->offsetInBytes;
        Rpp8u *temp_in = (Rpp8u *)pSrc;

        Rpp32u elementsInRow = oSrcSize.width * srcDescPtr->c;

        for (int j = 0; j < oSrcSize.height; j++)
        {
            hipMemcpy(offsetted_input, temp_in, elementsInRow * sizeof (Rpp8u), hipMemcpyDeviceToDevice);
            temp_in += elementsInRow;
            offsetted_input += srcDescPtr->strides.hStride;
        }
		
        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        status = rppt_resize_gpu(d_input, srcDescPtr, d_output, dstDescPtr, dstImgSizes, interpolationType, roiTensorPtrSrc, roiTypeSrc, handle);
        hipDeviceSynchronize();
      
        rppDestroyGPU(handle);
        Rpp32u elementsInRowMax = dstDescPtr->w * dstDescPtr->c;
        Rpp32u elementsInRowout = dstImgSizes->width * dstDescPtr->c;
        //Rpp32u elementsInRowout = oSrcSize.width * dstDescPtr->c;
        offsetted_output = d_output + dstDescPtr->offsetInBytes;
        Rpp8u *temp_output = pDst;
        for (int k = 0; k < dstImgSizes->height; k++)
        //for (int k = 0; k < oSrcSize.height; k++)
        {
            hipMemcpy(temp_output, offsetted_output, elementsInRowout * sizeof (Rpp8u),hipMemcpyDeviceToDevice);
            temp_output += elementsInRowout;
            offsetted_output += elementsInRowMax;
	    }

        hipFree(d_input);
        hipFree(d_output);
	    hipHostFree(roiTensorPtrSrc);
	    hipHostFree(dstImgSizes);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiResize_8u_C1R_Ctx(const Npp8u *pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI, Npp8u *pDst, int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI, int eInterpolation, NppStreamContext nppStreamCtx)
{
        //pln
        int noOfImages = 1;
        int ip_channel = 1;
        RpptDesc srcDesc, dstDesc;
        RpptDescPtr srcDescPtr, dstDescPtr;
        srcDescPtr = &srcDesc;
        dstDescPtr = &dstDesc;

        srcDescPtr->layout = RpptLayout::NCHW;
        dstDescPtr->layout = RpptLayout::NCHW;
        srcDescPtr->dataType = RpptDataType::U8;
        dstDescPtr->dataType = RpptDataType::U8;

        srcDescPtr->numDims = 4;
        dstDescPtr->numDims = 4;

        srcDescPtr->offsetInBytes = 64;
        dstDescPtr->offsetInBytes = 0;

        srcDescPtr->n = noOfImages;
        srcDescPtr->h = oSrcSize.height;
        srcDescPtr->w = oSrcSize.width;
        srcDescPtr->c = ip_channel;

        dstDescPtr->n = noOfImages;
        dstDescPtr->h = oSrcSize.height;
        dstDescPtr->w = oSrcSize.width;
        dstDescPtr->c = ip_channel;

        RppStatus status;
        RpptInterpolationType interpolationType = RpptInterpolationType::BILINEAR;

        if(eInterpolation == NPPI_INTER_UNDEFINED) {
                interpolationType = RpptInterpolationType::NEAREST_NEIGHBOR;
        } else if (eInterpolation == NPPI_INTER_LINEAR) {
                interpolationType = RpptInterpolationType::BILINEAR;
        } else if (eInterpolation == NPPI_INTER_CUBIC) {
                interpolationType = RpptInterpolationType::BICUBIC;
        } else if(eInterpolation == NPPI_INTER_LANCZOS) {
                interpolationType = RpptInterpolationType::LANCZOS;
        } else {
                status = RPP_ERROR_INVALID_ARGUMENTS;
                return(hipRppStatusTocudaNppStatus(status));
        }

    // Optionally set w stride as a multiple of 8 for src/dst

        srcDescPtr->w = ((srcDescPtr->w / 8) * 8) + 8;
        dstDescPtr->w = ((dstDescPtr->w / 8) * 8) + 8;

    // Set n/c/h/w strides for src/dst
        RpptROI *roiTensorPtrSrc;
        hipHostMalloc(&roiTensorPtrSrc, noOfImages * sizeof(RpptROI));
        roiTensorPtrSrc[0].xywhROI.xy.x = 0;
        roiTensorPtrSrc[0].xywhROI.xy.y = 0;
        roiTensorPtrSrc[0].xywhROI.roiWidth = oSrcSize.width;
        roiTensorPtrSrc[0].xywhROI.roiHeight = oSrcSize.height;
        RpptRoiType roiTypeSrc;
        roiTypeSrc = RpptRoiType::XYWH;
        RpptImagePatch *dstImgSizes;
        hipHostMalloc(&dstImgSizes, noOfImages * sizeof(RpptImagePatch));
        dstImgSizes[0].width = oDstSize.width;
        dstImgSizes[0].height = oDstSize.height;

        srcDescPtr->strides.nStride = ip_channel * srcDescPtr->w * srcDescPtr->h;
	    srcDescPtr->strides.cStride = srcDescPtr->w * srcDescPtr->h;
	    srcDescPtr->strides.hStride = srcDescPtr->w;
	    srcDescPtr->strides.wStride = 1;
		
	    dstDescPtr->strides.nStride = ip_channel * dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.cStride = dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.hStride = dstDescPtr->w;
        dstDescPtr->strides.wStride = 1;

        unsigned long long ioBufferSize = 0;
        unsigned long long oBufferSize = 0;

        ioBufferSize = (unsigned long long)srcDescPtr->h * (unsigned long long)srcDescPtr->w * (unsigned long long)srcDescPtr->c * (unsigned long long)noOfImages;
        oBufferSize = (unsigned long long)dstDescPtr->h * (unsigned long long)dstDescPtr->w * (unsigned long long)dstDescPtr->c * (unsigned long long)noOfImages;

        unsigned long long ioBufferSizeInBytes_u8 = ioBufferSize + srcDescPtr->offsetInBytes;
        unsigned long long oBufferSizeInBytes_u8 = oBufferSize + dstDescPtr->offsetInBytes;

        Rpp8u *d_input,*d_output;
        hipMalloc(&d_input,ioBufferSizeInBytes_u8);
        hipMalloc(&d_output,oBufferSizeInBytes_u8);
        Rpp8u *offsetted_input, *offsetted_output;
        offsetted_input = d_input + srcDescPtr->offsetInBytes;
        Rpp8u *temp_in = (Rpp8u *)pSrc;

        Rpp32u elementsInRow = oSrcSize.width * srcDescPtr->c;

        for (int j = 0; j < oSrcSize.height; j++)
        {
            hipMemcpy(offsetted_input, temp_in, elementsInRow * sizeof (Rpp8u), hipMemcpyDeviceToDevice);
            temp_in += elementsInRow;
            offsetted_input += srcDescPtr->strides.hStride;
        }

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        status = rppt_resize_gpu(d_input, srcDescPtr, d_output, dstDescPtr, dstImgSizes, interpolationType, roiTensorPtrSrc, roiTypeSrc, handle);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        Rpp32u elementsInRowMax = dstDescPtr->w * dstDescPtr->c;
        Rpp32u elementsInRowout = dstImgSizes->width * dstDescPtr->c;
        //Rpp32u elementsInRowout = oSrcSize.width * dstDescPtr->c;
        offsetted_output = d_output + dstDescPtr->offsetInBytes;
        Rpp8u *temp_output = pDst;
        for (int k = 0; k < dstImgSizes->height; k++)
        //for (int k = 0; k < oSrcSize.height; k++)
        {
            hipMemcpy(temp_output, offsetted_output, elementsInRowout * sizeof (Rpp8u),hipMemcpyDeviceToDevice);
            temp_output += elementsInRowout;
            offsetted_output += elementsInRowMax;
        }

        hipFree(d_input);
        hipFree(d_output);
        hipHostFree(roiTensorPtrSrc);
        hipHostFree(dstImgSizes);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiResize_8u_C1R(const Npp8u *pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI, Npp8u *pDst, int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI, int eInterpolation)
{
        //pln
        int noOfImages = 1;
        int ip_channel = 1;
        RpptDesc srcDesc, dstDesc;
        RpptDescPtr srcDescPtr, dstDescPtr;
        srcDescPtr = &srcDesc;
        dstDescPtr = &dstDesc;

        srcDescPtr->layout = RpptLayout::NCHW;
        dstDescPtr->layout = RpptLayout::NCHW;
        srcDescPtr->dataType = RpptDataType::U8;
        dstDescPtr->dataType = RpptDataType::U8;

        srcDescPtr->numDims = 4;
        dstDescPtr->numDims = 4;

        srcDescPtr->offsetInBytes = 64;
        dstDescPtr->offsetInBytes = 0;

        srcDescPtr->n = noOfImages;
        srcDescPtr->h = oSrcSize.height;
        srcDescPtr->w = oSrcSize.width;
        srcDescPtr->c = ip_channel;

        dstDescPtr->n = noOfImages;
        dstDescPtr->h = oSrcSize.height;
        dstDescPtr->w = oSrcSize.width;
        dstDescPtr->c = ip_channel;

        RppStatus status;
        RpptInterpolationType interpolationType = RpptInterpolationType::BILINEAR;

        if(eInterpolation == NPPI_INTER_UNDEFINED) {
                interpolationType = RpptInterpolationType::NEAREST_NEIGHBOR;
        } else if (eInterpolation == NPPI_INTER_LINEAR) {
                interpolationType = RpptInterpolationType::BILINEAR;
        } else if (eInterpolation == NPPI_INTER_CUBIC) {
                interpolationType = RpptInterpolationType::BICUBIC;
        } else if(eInterpolation == NPPI_INTER_LANCZOS) {
                interpolationType = RpptInterpolationType::LANCZOS;
        } else {
                status = RPP_ERROR_INVALID_ARGUMENTS;
                return(hipRppStatusTocudaNppStatus(status));
        }

    // Optionally set w stride as a multiple of 8 for src/dst

        srcDescPtr->w = ((srcDescPtr->w / 8) * 8) + 8;
        dstDescPtr->w = ((dstDescPtr->w / 8) * 8) + 8;

    // Set n/c/h/w strides for src/dst
        RpptROI *roiTensorPtrSrc;
        hipHostMalloc(&roiTensorPtrSrc, noOfImages * sizeof(RpptROI));
        roiTensorPtrSrc[0].xywhROI.xy.x = 0;
        roiTensorPtrSrc[0].xywhROI.xy.y = 0;
        roiTensorPtrSrc[0].xywhROI.roiWidth = oSrcSize.width;
        roiTensorPtrSrc[0].xywhROI.roiHeight = oSrcSize.height;
        RpptRoiType roiTypeSrc;
        roiTypeSrc = RpptRoiType::XYWH;
        RpptImagePatch *dstImgSizes;
        hipHostMalloc(&dstImgSizes, noOfImages * sizeof(RpptImagePatch));
        dstImgSizes[0].width = oDstSize.width;
        dstImgSizes[0].height = oDstSize.height;

        srcDescPtr->strides.nStride = ip_channel * srcDescPtr->w * srcDescPtr->h;
	    srcDescPtr->strides.cStride = srcDescPtr->w * srcDescPtr->h;
	    srcDescPtr->strides.hStride = srcDescPtr->w;
	    srcDescPtr->strides.wStride = 1;
		
	    dstDescPtr->strides.nStride = ip_channel * dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.cStride = dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.hStride = dstDescPtr->w;
        dstDescPtr->strides.wStride = 1;

        unsigned long long ioBufferSize = 0;
        unsigned long long oBufferSize = 0;

        ioBufferSize = (unsigned long long)srcDescPtr->h * (unsigned long long)srcDescPtr->w * (unsigned long long)srcDescPtr->c * (unsigned long long)noOfImages;
        oBufferSize = (unsigned long long)dstDescPtr->h * (unsigned long long)dstDescPtr->w * (unsigned long long)dstDescPtr->c * (unsigned long long)noOfImages;

        unsigned long long ioBufferSizeInBytes_u8 = ioBufferSize + srcDescPtr->offsetInBytes;
        unsigned long long oBufferSizeInBytes_u8 = oBufferSize + dstDescPtr->offsetInBytes;

        Rpp8u *d_input,*d_output;
        hipMalloc(&d_input,ioBufferSizeInBytes_u8);
        hipMalloc(&d_output,oBufferSizeInBytes_u8);
        Rpp8u *offsetted_input, *offsetted_output;
        offsetted_input = d_input + srcDescPtr->offsetInBytes;
        Rpp8u *temp_in = (Rpp8u *)pSrc;

        Rpp32u elementsInRow = oSrcSize.width * srcDescPtr->c;

        for (int j = 0; j < oSrcSize.height; j++)
        {
            hipMemcpy(offsetted_input, temp_in, elementsInRow * sizeof (Rpp8u), hipMemcpyDeviceToDevice);
            temp_in += elementsInRow;
            offsetted_input += srcDescPtr->strides.hStride;
        }

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        status = rppt_resize_gpu(d_input, srcDescPtr, d_output, dstDescPtr, dstImgSizes, interpolationType, roiTensorPtrSrc, roiTypeSrc, handle);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        Rpp32u elementsInRowMax = dstDescPtr->w * dstDescPtr->c;
        Rpp32u elementsInRowout = dstImgSizes->width * dstDescPtr->c;
        //Rpp32u elementsInRowout = oSrcSize.width * dstDescPtr->c;
        offsetted_output = d_output + dstDescPtr->offsetInBytes;
        Rpp8u *temp_output = pDst;
        for (int k = 0; k < dstImgSizes->height; k++)
        //for (int k = 0; k < oSrcSize.height; k++)
        {
            hipMemcpy(temp_output, offsetted_output, elementsInRowout * sizeof (Rpp8u),hipMemcpyDeviceToDevice);
            temp_output += elementsInRowout;
            offsetted_output += elementsInRowMax;
        }

        hipFree(d_input);
        hipFree(d_output);
        hipHostFree(roiTensorPtrSrc);
        hipHostFree(dstImgSizes);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiResize_8u_C3R_Ctx(const Npp8u *pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI, Npp8u *pDst, int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI, int eInterpolation, NppStreamContext nppStreamCtx)
{
        //pkd
        int noOfImages = 1;
        int ip_channel = 3;
        RpptDesc srcDesc, dstDesc;
        RpptDescPtr srcDescPtr, dstDescPtr;
        srcDescPtr = &srcDesc;
        dstDescPtr = &dstDesc;

        srcDescPtr->layout = RpptLayout::NHWC;
        dstDescPtr->layout = RpptLayout::NHWC;
        srcDescPtr->dataType = RpptDataType::U8;
        dstDescPtr->dataType = RpptDataType::U8;

        srcDescPtr->numDims = 4;
        dstDescPtr->numDims = 4;

        srcDescPtr->offsetInBytes = 64;
        dstDescPtr->offsetInBytes = 0;

        srcDescPtr->n = noOfImages;
        srcDescPtr->h = oSrcSize.height;
        srcDescPtr->w = oSrcSize.width;
        srcDescPtr->c = ip_channel;

        dstDescPtr->n = noOfImages;
        dstDescPtr->h = oSrcSize.height;
        dstDescPtr->w = oSrcSize.width;
        dstDescPtr->c = ip_channel;

        RppStatus status;
        RpptInterpolationType interpolationType = RpptInterpolationType::BILINEAR;

        if(eInterpolation == NPPI_INTER_UNDEFINED) {
            interpolationType = RpptInterpolationType::NEAREST_NEIGHBOR;
        } else if (eInterpolation == NPPI_INTER_LINEAR) {
            interpolationType = RpptInterpolationType::BILINEAR;
        } else if (eInterpolation == NPPI_INTER_CUBIC) {
            interpolationType = RpptInterpolationType::BICUBIC;
        } else if(eInterpolation == NPPI_INTER_LANCZOS) {
            interpolationType = RpptInterpolationType::LANCZOS;
        } else {
            status = RPP_ERROR_INVALID_ARGUMENTS;
            return(hipRppStatusTocudaNppStatus(status));
        }

    // Optionally set w stride as a multiple of 8 for src/dst

        srcDescPtr->w = ((srcDescPtr->w / 8) * 8) + 8;
        dstDescPtr->w = ((dstDescPtr->w / 8) * 8) + 8;

    // Set n/c/h/w strides for src/dst
        RpptROI *roiTensorPtrSrc;
        hipHostMalloc(&roiTensorPtrSrc, noOfImages * sizeof(RpptROI));
        roiTensorPtrSrc[0].xywhROI.xy.x = 0;
        roiTensorPtrSrc[0].xywhROI.xy.y = 0;
        roiTensorPtrSrc[0].xywhROI.roiWidth = oSrcSize.width;
        roiTensorPtrSrc[0].xywhROI.roiHeight = oSrcSize.height;
        RpptRoiType roiTypeSrc;
        roiTypeSrc = RpptRoiType::XYWH;
        RpptImagePatch *dstImgSizes;
        hipHostMalloc(&dstImgSizes, noOfImages * sizeof(RpptImagePatch));
        dstImgSizes[0].width = oDstSize.width;
        dstImgSizes[0].height = oDstSize.height;

        srcDescPtr->strides.nStride = srcDescPtr->c * srcDescPtr->w * srcDescPtr->h;
        srcDescPtr->strides.hStride = srcDescPtr->c * srcDescPtr->w;
        srcDescPtr->strides.wStride = srcDescPtr->c;
        srcDescPtr->strides.cStride = 1;

	    dstDescPtr->strides.nStride = dstDescPtr->c * dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.hStride = dstDescPtr->c * dstDescPtr->w;
        dstDescPtr->strides.wStride = dstDescPtr->c;
        dstDescPtr->strides.cStride = 1;

        unsigned long long ioBufferSize = 0;
        unsigned long long oBufferSize = 0;

        ioBufferSize = (unsigned long long)srcDescPtr->h * (unsigned long long)srcDescPtr->w * (unsigned long long)srcDescPtr->c * (unsigned long long)noOfImages;
        oBufferSize = (unsigned long long)dstDescPtr->h * (unsigned long long)dstDescPtr->w * (unsigned long long)dstDescPtr->c * (unsigned long long)noOfImages;

        unsigned long long ioBufferSizeInBytes_u8 = ioBufferSize + srcDescPtr->offsetInBytes;
        unsigned long long oBufferSizeInBytes_u8 = oBufferSize + dstDescPtr->offsetInBytes;

        Rpp8u *d_input,*d_output;
        hipMalloc(&d_input,ioBufferSizeInBytes_u8);
        hipMalloc(&d_output,oBufferSizeInBytes_u8);
        Rpp8u *offsetted_input, *offsetted_output;
        offsetted_input = d_input + srcDescPtr->offsetInBytes;
        Rpp8u *temp_in = (Rpp8u *)pSrc;

        Rpp32u elementsInRow = oSrcSize.width * srcDescPtr->c;

        for (int j = 0; j < oSrcSize.height; j++)
        {
            hipMemcpy(offsetted_input, temp_in, elementsInRow * sizeof (Rpp8u), hipMemcpyDeviceToDevice);
            temp_in += elementsInRow;
            offsetted_input += srcDescPtr->strides.hStride;
        }

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        status = rppt_resize_gpu(d_input, srcDescPtr, d_output, dstDescPtr, dstImgSizes, interpolationType, roiTensorPtrSrc, roiTypeSrc, handle);
	    hipDeviceSynchronize();

        rppDestroyGPU(handle);
        Rpp32u elementsInRowMax = dstDescPtr->w * dstDescPtr->c;
        Rpp32u elementsInRowout = dstImgSizes->width * dstDescPtr->c;
        //Rpp32u elementsInRowout = oSrcSize.width * dstDescPtr->c;
        offsetted_output = d_output + dstDescPtr->offsetInBytes;
        Rpp8u *temp_output = pDst;
        for (int k = 0; k < dstImgSizes->height; k++)
        //for (int k = 0; k < oSrcSize.height; k++)
        {
            hipMemcpy(temp_output, offsetted_output, elementsInRowout * sizeof (Rpp8u),hipMemcpyDeviceToDevice);
            temp_output += elementsInRowout;
            offsetted_output += elementsInRowMax;
        }

        hipFree(d_input);
        hipFree(d_output);
        hipHostFree(roiTensorPtrSrc);
        hipHostFree(dstImgSizes);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiResize_16f_C1R_Ctx(const Npp16f *pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI, Npp16f *pDst, int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI, int eInterpolation, NppStreamContext nppStreamCtx)
{
        //pln
        int noOfImages = 1;
        int ip_channel = 1;
        RpptDesc srcDesc, dstDesc;
        RpptDescPtr srcDescPtr, dstDescPtr;
        srcDescPtr = &srcDesc;
        dstDescPtr = &dstDesc;

        srcDescPtr->layout = RpptLayout::NCHW;
        dstDescPtr->layout = RpptLayout::NCHW;
        srcDescPtr->dataType = RpptDataType::F16;
        dstDescPtr->dataType = RpptDataType::F16;

        srcDescPtr->numDims = 4;
        dstDescPtr->numDims = 4;

        srcDescPtr->offsetInBytes = 64;
        dstDescPtr->offsetInBytes = 0;

        srcDescPtr->n = noOfImages;
        srcDescPtr->h = oSrcSize.height;
        srcDescPtr->w = oSrcSize.width;
        srcDescPtr->c = ip_channel;

        dstDescPtr->n = noOfImages;
        dstDescPtr->h = oSrcSize.height;
        dstDescPtr->w = oSrcSize.width;
        dstDescPtr->c = ip_channel;

        RppStatus status;
        RpptInterpolationType interpolationType = RpptInterpolationType::BILINEAR;

        if(eInterpolation == NPPI_INTER_UNDEFINED) {
            interpolationType = RpptInterpolationType::NEAREST_NEIGHBOR;
        } else if (eInterpolation == NPPI_INTER_LINEAR) {
            interpolationType = RpptInterpolationType::BILINEAR;
        } else if (eInterpolation == NPPI_INTER_CUBIC) {
            interpolationType = RpptInterpolationType::BICUBIC;
        } else if(eInterpolation == NPPI_INTER_LANCZOS) {
            interpolationType = RpptInterpolationType::LANCZOS;
        } else {
            status = RPP_ERROR_INVALID_ARGUMENTS;
            return(hipRppStatusTocudaNppStatus(status));
        }

    // Optionally set w stride as a multiple of 8 for src/dst

        srcDescPtr->w = ((srcDescPtr->w / 8) * 8) + 8;
        dstDescPtr->w = ((dstDescPtr->w / 8) * 8) + 8;

    // Set n/c/h/w strides for src/dst
        RpptROI *roiTensorPtrSrc;
        hipHostMalloc(&roiTensorPtrSrc, noOfImages * sizeof(RpptROI));
        roiTensorPtrSrc[0].xywhROI.xy.x = 0;
        roiTensorPtrSrc[0].xywhROI.xy.y = 0;
        roiTensorPtrSrc[0].xywhROI.roiWidth = oSrcSize.width;
        roiTensorPtrSrc[0].xywhROI.roiHeight = oSrcSize.height;
        RpptRoiType roiTypeSrc;
        roiTypeSrc = RpptRoiType::XYWH;
        RpptImagePatch *dstImgSizes;
        hipHostMalloc(&dstImgSizes, noOfImages * sizeof(RpptImagePatch));
        dstImgSizes[0].width = oDstSize.width;
        dstImgSizes[0].height = oDstSize.height;

        srcDescPtr->strides.nStride = ip_channel * srcDescPtr->w * srcDescPtr->h;
        srcDescPtr->strides.cStride = srcDescPtr->w * srcDescPtr->h;
        srcDescPtr->strides.hStride = srcDescPtr->w;
        srcDescPtr->strides.wStride = 1;

        dstDescPtr->strides.nStride = ip_channel * dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.cStride = dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.hStride = dstDescPtr->w;
        dstDescPtr->strides.wStride = 1;

        unsigned long long ioBufferSize = 0;
        unsigned long long oBufferSize = 0;

        ioBufferSize = (unsigned long long)srcDescPtr->h * (unsigned long long)srcDescPtr->w * (unsigned long long)srcDescPtr->c * (unsigned long long)noOfImages;
        oBufferSize = (unsigned long long)dstDescPtr->h * (unsigned long long)dstDescPtr->w * (unsigned long long)dstDescPtr->c * (unsigned long long)noOfImages;

        unsigned long long ioBufferSizeInBytes_f16 = (ioBufferSize * 2) + srcDescPtr->offsetInBytes;
	    unsigned long long oBufferSizeInBytes_f16 = (oBufferSize * 2) + dstDescPtr->offsetInBytes;

        Rpp16f *d_input,*d_output;
        hipMalloc(&d_input,ioBufferSizeInBytes_f16);
        hipMalloc(&d_output,oBufferSizeInBytes_f16);
        Rpp16f *offsetted_input, *offsetted_output;
        offsetted_input = d_input + srcDescPtr->offsetInBytes;
        Rpp16f *temp_in = (Rpp16f *)pSrc;
		
        Rpp32u elementsInRow = oSrcSize.width * srcDescPtr->c;

        for (int j = 0; j < oSrcSize.height; j++)
        {
            hipMemcpy(offsetted_input, temp_in, elementsInRow * sizeof (Rpp16f), hipMemcpyDeviceToDevice);
            temp_in += elementsInRow;
            offsetted_input += srcDescPtr->strides.hStride;
        }

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        status = rppt_resize_gpu(d_input, srcDescPtr, d_output, dstDescPtr, dstImgSizes, interpolationType, roiTensorPtrSrc, roiTypeSrc, handle);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        Rpp32u elementsInRowMax = dstDescPtr->w * dstDescPtr->c;
        Rpp32u elementsInRowout = dstImgSizes->width * dstDescPtr->c;
        //Rpp32u elementsInRowout = oSrcSize.width * dstDescPtr->c;
        offsetted_output = d_output + dstDescPtr->offsetInBytes;
        Rpp16f *temp_output = (Rpp16f *)pDst;
        for (int k = 0; k < dstImgSizes->height; k++)
        //for (int k = 0; k < oSrcSize.height; k++)
        {
            hipMemcpy(temp_output, offsetted_output, elementsInRowout * sizeof (Rpp16f),hipMemcpyDeviceToDevice);
            temp_output += elementsInRowout;
            offsetted_output += elementsInRowMax;
        }

        hipFree(d_input);
        hipFree(d_output);
        hipHostFree(roiTensorPtrSrc);
        hipHostFree(dstImgSizes);

        return(hipRppStatusTocudaNppStatus(status));
}
		
NppStatus nppiResize_16f_C1R(const Npp16f *pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI, Npp16f *pDst, int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI, int eInterpolation)
{
        //pln
        int noOfImages = 1;
        int ip_channel = 1;
        RpptDesc srcDesc, dstDesc;
        RpptDescPtr srcDescPtr, dstDescPtr;
        srcDescPtr = &srcDesc;
        dstDescPtr = &dstDesc;

        srcDescPtr->layout = RpptLayout::NCHW;
        dstDescPtr->layout = RpptLayout::NCHW;
        srcDescPtr->dataType = RpptDataType::F16;
        dstDescPtr->dataType = RpptDataType::F16;

        srcDescPtr->numDims = 4;
        dstDescPtr->numDims = 4;

        srcDescPtr->offsetInBytes = 64;
        dstDescPtr->offsetInBytes = 0;

        srcDescPtr->n = noOfImages;
        srcDescPtr->h = oSrcSize.height;
        srcDescPtr->w = oSrcSize.width;
        srcDescPtr->c = ip_channel;

        dstDescPtr->n = noOfImages;
        dstDescPtr->h = oSrcSize.height;
        dstDescPtr->w = oSrcSize.width;
        dstDescPtr->c = ip_channel;

        RppStatus status;
        RpptInterpolationType interpolationType = RpptInterpolationType::BILINEAR;

        if(eInterpolation == NPPI_INTER_UNDEFINED) {
            interpolationType = RpptInterpolationType::NEAREST_NEIGHBOR;
        } else if (eInterpolation == NPPI_INTER_LINEAR) {
            interpolationType = RpptInterpolationType::BILINEAR;
        } else if (eInterpolation == NPPI_INTER_CUBIC) {
            interpolationType = RpptInterpolationType::BICUBIC;
        } else if(eInterpolation == NPPI_INTER_LANCZOS) {
            interpolationType = RpptInterpolationType::LANCZOS;
        } else {
            status = RPP_ERROR_INVALID_ARGUMENTS;
            return(hipRppStatusTocudaNppStatus(status));
        }

    // Optionally set w stride as a multiple of 8 for src/dst

        srcDescPtr->w = ((srcDescPtr->w / 8) * 8) + 8;
        dstDescPtr->w = ((dstDescPtr->w / 8) * 8) + 8;

    // Set n/c/h/w strides for src/dst
        RpptROI *roiTensorPtrSrc;
        hipHostMalloc(&roiTensorPtrSrc, noOfImages * sizeof(RpptROI));
        roiTensorPtrSrc[0].xywhROI.xy.x = 0;
        roiTensorPtrSrc[0].xywhROI.xy.y = 0;
        roiTensorPtrSrc[0].xywhROI.roiWidth = oSrcSize.width;
        roiTensorPtrSrc[0].xywhROI.roiHeight = oSrcSize.height;
        RpptRoiType roiTypeSrc;
        roiTypeSrc = RpptRoiType::XYWH;
        RpptImagePatch *dstImgSizes;
        hipHostMalloc(&dstImgSizes, noOfImages * sizeof(RpptImagePatch));
        dstImgSizes[0].width = oDstSize.width;
        dstImgSizes[0].height = oDstSize.height;

        srcDescPtr->strides.nStride = ip_channel * srcDescPtr->w * srcDescPtr->h;
        srcDescPtr->strides.cStride = srcDescPtr->w * srcDescPtr->h;
        srcDescPtr->strides.hStride = srcDescPtr->w;
        srcDescPtr->strides.wStride = 1;

        dstDescPtr->strides.nStride = ip_channel * dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.cStride = dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.hStride = dstDescPtr->w;
        dstDescPtr->strides.wStride = 1;

        unsigned long long ioBufferSize = 0;
        unsigned long long oBufferSize = 0;

        ioBufferSize = (unsigned long long)srcDescPtr->h * (unsigned long long)srcDescPtr->w * (unsigned long long)srcDescPtr->c * (unsigned long long)noOfImages;
        oBufferSize = (unsigned long long)dstDescPtr->h * (unsigned long long)dstDescPtr->w * (unsigned long long)dstDescPtr->c * (unsigned long long)noOfImages;

        unsigned long long ioBufferSizeInBytes_f16 = (ioBufferSize * 2) + srcDescPtr->offsetInBytes;
	    unsigned long long oBufferSizeInBytes_f16 = (oBufferSize * 2) + dstDescPtr->offsetInBytes;

        Rpp16f *d_input,*d_output;
        hipMalloc(&d_input,ioBufferSizeInBytes_f16);
        hipMalloc(&d_output,oBufferSizeInBytes_f16);
        Rpp16f *offsetted_input, *offsetted_output;
        offsetted_input = d_input + srcDescPtr->offsetInBytes;
        Rpp16f *temp_in = (Rpp16f *)pSrc;
		
        Rpp32u elementsInRow = oSrcSize.width * srcDescPtr->c;

        for (int j = 0; j < oSrcSize.height; j++)
        {
            hipMemcpy(offsetted_input, temp_in, elementsInRow * sizeof (Rpp16f), hipMemcpyDeviceToDevice);
            temp_in += elementsInRow;
            offsetted_input += srcDescPtr->strides.hStride;
        }

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        status = rppt_resize_gpu(d_input, srcDescPtr, d_output, dstDescPtr, dstImgSizes, interpolationType, roiTensorPtrSrc, roiTypeSrc, handle);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        Rpp32u elementsInRowMax = dstDescPtr->w * dstDescPtr->c;
        Rpp32u elementsInRowout = dstImgSizes->width * dstDescPtr->c;
        //Rpp32u elementsInRowout = oSrcSize.width * dstDescPtr->c;
        offsetted_output = d_output + dstDescPtr->offsetInBytes;
        Rpp16f *temp_output = (Rpp16f *)pDst;
        for (int k = 0; k < dstImgSizes->height; k++)
        //for (int k = 0; k < oSrcSize.height; k++)
        {
            hipMemcpy(temp_output, offsetted_output, elementsInRowout * sizeof (Rpp16f),hipMemcpyDeviceToDevice);
            temp_output += elementsInRowout;
            offsetted_output += elementsInRowMax;
        }

        hipFree(d_input);
        hipFree(d_output);
        hipHostFree(roiTensorPtrSrc);
        hipHostFree(dstImgSizes);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiResize_16f_C3R_Ctx(const Npp16f *pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI, Npp16f *pDst, int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI, int eInterpolation, NppStreamContext nppStreamCtx)
{
        //pln
        int noOfImages = 1;
        int ip_channel = 3;
        RpptDesc srcDesc, dstDesc;
        RpptDescPtr srcDescPtr, dstDescPtr;
        srcDescPtr = &srcDesc;
        dstDescPtr = &dstDesc;

        srcDescPtr->layout = RpptLayout::NHWC;
        dstDescPtr->layout = RpptLayout::NHWC;
        srcDescPtr->dataType = RpptDataType::F16;
        dstDescPtr->dataType = RpptDataType::F16;

        srcDescPtr->numDims = 4;
        dstDescPtr->numDims = 4;

        srcDescPtr->offsetInBytes = 64;
        dstDescPtr->offsetInBytes = 0;

        srcDescPtr->n = noOfImages;
        srcDescPtr->h = oSrcSize.height;
        srcDescPtr->w = oSrcSize.width;
        srcDescPtr->c = ip_channel;

        dstDescPtr->n = noOfImages;
        dstDescPtr->h = oSrcSize.height;
        dstDescPtr->w = oSrcSize.width;
        dstDescPtr->c = ip_channel;

        RppStatus status;
        RpptInterpolationType interpolationType = RpptInterpolationType::BILINEAR;

        if(eInterpolation == NPPI_INTER_UNDEFINED) {
            interpolationType = RpptInterpolationType::NEAREST_NEIGHBOR;
        } else if (eInterpolation == NPPI_INTER_LINEAR) {
            interpolationType = RpptInterpolationType::BILINEAR;
        } else if (eInterpolation == NPPI_INTER_CUBIC) {
            interpolationType = RpptInterpolationType::BICUBIC;
        } else if(eInterpolation == NPPI_INTER_LANCZOS) {
            interpolationType = RpptInterpolationType::LANCZOS;
        } else {
            status = RPP_ERROR_INVALID_ARGUMENTS;
            return(hipRppStatusTocudaNppStatus(status));
        }

    // Optionally set w stride as a multiple of 8 for src/dst

        srcDescPtr->w = ((srcDescPtr->w / 8) * 8) + 8;
        dstDescPtr->w = ((dstDescPtr->w / 8) * 8) + 8;

    // Set n/c/h/w strides for src/dst
        RpptROI *roiTensorPtrSrc;
        hipHostMalloc(&roiTensorPtrSrc, noOfImages * sizeof(RpptROI));
        roiTensorPtrSrc[0].xywhROI.xy.x = 0;
        roiTensorPtrSrc[0].xywhROI.xy.y = 0;
        roiTensorPtrSrc[0].xywhROI.roiWidth = oSrcSize.width;
        roiTensorPtrSrc[0].xywhROI.roiHeight = oSrcSize.height;
        RpptRoiType roiTypeSrc;
        roiTypeSrc = RpptRoiType::XYWH;
        RpptImagePatch *dstImgSizes;
        hipHostMalloc(&dstImgSizes, noOfImages * sizeof(RpptImagePatch));
        dstImgSizes[0].width = oDstSize.width;
        dstImgSizes[0].height = oDstSize.height;

        srcDescPtr->strides.nStride = srcDescPtr->c * srcDescPtr->w * srcDescPtr->h;
        srcDescPtr->strides.hStride = srcDescPtr->c * srcDescPtr->w;
        srcDescPtr->strides.wStride = srcDescPtr->c;
        srcDescPtr->strides.cStride = 1;

        dstDescPtr->strides.nStride = dstDescPtr->c * dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.hStride = dstDescPtr->c * dstDescPtr->w;
        dstDescPtr->strides.wStride = dstDescPtr->c;
        dstDescPtr->strides.cStride = 1;

        unsigned long long ioBufferSize = 0;
        unsigned long long oBufferSize = 0;

        ioBufferSize = (unsigned long long)srcDescPtr->h * (unsigned long long)srcDescPtr->w * (unsigned long long)srcDescPtr->c * (unsigned long long)noOfImages;
        oBufferSize = (unsigned long long)dstDescPtr->h * (unsigned long long)dstDescPtr->w * (unsigned long long)dstDescPtr->c * (unsigned long long)noOfImages;

        unsigned long long ioBufferSizeInBytes_f16 = (ioBufferSize * 2) + srcDescPtr->offsetInBytes;
	    unsigned long long oBufferSizeInBytes_f16 = (oBufferSize * 2) + dstDescPtr->offsetInBytes;

        Rpp16f *d_input,*d_output;
        hipMalloc(&d_input,ioBufferSizeInBytes_f16);
        hipMalloc(&d_output,oBufferSizeInBytes_f16);
        Rpp16f *offsetted_input, *offsetted_output;
        offsetted_input = d_input + srcDescPtr->offsetInBytes;
        Rpp16f *temp_in = (Rpp16f *)pSrc;
		
        Rpp32u elementsInRow = oSrcSize.width * srcDescPtr->c;

        for (int j = 0; j < oSrcSize.height; j++)
        {
            hipMemcpy(offsetted_input, temp_in, elementsInRow * sizeof (Rpp16f), hipMemcpyDeviceToDevice);
            temp_in += elementsInRow;
            offsetted_input += srcDescPtr->strides.hStride;
        }

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        status = rppt_resize_gpu(d_input, srcDescPtr, d_output, dstDescPtr, dstImgSizes, interpolationType, roiTensorPtrSrc, roiTypeSrc, handle);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        Rpp32u elementsInRowMax = dstDescPtr->w * dstDescPtr->c;
        Rpp32u elementsInRowout = dstImgSizes->width * dstDescPtr->c;
        //Rpp32u elementsInRowout = oSrcSize.width * dstDescPtr->c;
        offsetted_output = d_output + dstDescPtr->offsetInBytes;
        Rpp16f *temp_output = (Rpp16f *)pDst;
        for (int k = 0; k < dstImgSizes->height; k++)
        //for (int k = 0; k < oSrcSize.height; k++)
        {
            hipMemcpy(temp_output, offsetted_output, elementsInRowout * sizeof (Rpp16f),hipMemcpyDeviceToDevice);
            temp_output += elementsInRowout;
            offsetted_output += elementsInRowMax;
        }

        hipFree(d_input);
        hipFree(d_output);
        hipHostFree(roiTensorPtrSrc);
        hipHostFree(dstImgSizes);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiResize_16f_C3R(const Npp16f *pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI, Npp16f *pDst, int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI, int eInterpolation)
{
        //pln
        int noOfImages = 1;
        int ip_channel = 3;
        RpptDesc srcDesc, dstDesc;
        RpptDescPtr srcDescPtr, dstDescPtr;
        srcDescPtr = &srcDesc;
        dstDescPtr = &dstDesc;

        srcDescPtr->layout = RpptLayout::NHWC;
        dstDescPtr->layout = RpptLayout::NHWC;
        srcDescPtr->dataType = RpptDataType::F16;
        dstDescPtr->dataType = RpptDataType::F16;

        srcDescPtr->numDims = 4;
        dstDescPtr->numDims = 4;

        srcDescPtr->offsetInBytes = 64;
        dstDescPtr->offsetInBytes = 0;

        srcDescPtr->n = noOfImages;
        srcDescPtr->h = oSrcSize.height;
        srcDescPtr->w = oSrcSize.width;
        srcDescPtr->c = ip_channel;

        dstDescPtr->n = noOfImages;
        dstDescPtr->h = oSrcSize.height;
        dstDescPtr->w = oSrcSize.width;
        dstDescPtr->c = ip_channel;

        RppStatus status;
        RpptInterpolationType interpolationType = RpptInterpolationType::BILINEAR;

        if(eInterpolation == NPPI_INTER_UNDEFINED) {
                interpolationType = RpptInterpolationType::NEAREST_NEIGHBOR;
        } else if (eInterpolation == NPPI_INTER_LINEAR) {
                interpolationType = RpptInterpolationType::BILINEAR;
        } else if (eInterpolation == NPPI_INTER_CUBIC) {
                interpolationType = RpptInterpolationType::BICUBIC;
        } else if(eInterpolation == NPPI_INTER_LANCZOS) {
                interpolationType = RpptInterpolationType::LANCZOS;
        } else {
                status = RPP_ERROR_INVALID_ARGUMENTS;
                return(hipRppStatusTocudaNppStatus(status));
        }

    // Optionally set w stride as a multiple of 8 for src/dst

        srcDescPtr->w = ((srcDescPtr->w / 8) * 8) + 8;
        dstDescPtr->w = ((dstDescPtr->w / 8) * 8) + 8;

    // Set n/c/h/w strides for src/dst
        RpptROI *roiTensorPtrSrc;
        hipHostMalloc(&roiTensorPtrSrc, noOfImages * sizeof(RpptROI));
        roiTensorPtrSrc[0].xywhROI.xy.x = 0;
        roiTensorPtrSrc[0].xywhROI.xy.y = 0;
        roiTensorPtrSrc[0].xywhROI.roiWidth = oSrcSize.width;
        roiTensorPtrSrc[0].xywhROI.roiHeight = oSrcSize.height;
        RpptRoiType roiTypeSrc;
        roiTypeSrc = RpptRoiType::XYWH;
        RpptImagePatch *dstImgSizes;
        hipHostMalloc(&dstImgSizes, noOfImages * sizeof(RpptImagePatch));
        dstImgSizes[0].width = oDstSize.width;
        dstImgSizes[0].height = oDstSize.height;

        srcDescPtr->strides.nStride = srcDescPtr->c * srcDescPtr->w * srcDescPtr->h;
        srcDescPtr->strides.hStride = srcDescPtr->c * srcDescPtr->w;
        srcDescPtr->strides.wStride = srcDescPtr->c;
        srcDescPtr->strides.cStride = 1;

        dstDescPtr->strides.nStride = dstDescPtr->c * dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.hStride = dstDescPtr->c * dstDescPtr->w;
        dstDescPtr->strides.wStride = dstDescPtr->c;
        dstDescPtr->strides.cStride = 1;

        unsigned long long ioBufferSize = 0;
        unsigned long long oBufferSize = 0;

        ioBufferSize = (unsigned long long)srcDescPtr->h * (unsigned long long)srcDescPtr->w * (unsigned long long)srcDescPtr->c * (unsigned long long)noOfImages;
        oBufferSize = (unsigned long long)dstDescPtr->h * (unsigned long long)dstDescPtr->w * (unsigned long long)dstDescPtr->c * (unsigned long long)noOfImages;

        unsigned long long ioBufferSizeInBytes_f16 = (ioBufferSize * 2) + srcDescPtr->offsetInBytes;
	    unsigned long long oBufferSizeInBytes_f16 = (oBufferSize * 2) + dstDescPtr->offsetInBytes;

        Rpp16f *d_input,*d_output;
        hipMalloc(&d_input,ioBufferSizeInBytes_f16);
        hipMalloc(&d_output,oBufferSizeInBytes_f16);
        Rpp16f *offsetted_input, *offsetted_output;
        offsetted_input = d_input + srcDescPtr->offsetInBytes;
        Rpp16f *temp_in = (Rpp16f *)pSrc;
		
        Rpp32u elementsInRow = oSrcSize.width * srcDescPtr->c;

        for (int j = 0; j < oSrcSize.height; j++)
        {
            hipMemcpy(offsetted_input, temp_in, elementsInRow * sizeof (Rpp16f), hipMemcpyDeviceToDevice);
            temp_in += elementsInRow;
            offsetted_input += srcDescPtr->strides.hStride;
        }

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        status = rppt_resize_gpu(d_input, srcDescPtr, d_output, dstDescPtr, dstImgSizes, interpolationType, roiTensorPtrSrc, roiTypeSrc, handle);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        Rpp32u elementsInRowMax = dstDescPtr->w * dstDescPtr->c;
        Rpp32u elementsInRowout = dstImgSizes->width * dstDescPtr->c;
        //Rpp32u elementsInRowout = oSrcSize.width * dstDescPtr->c;
        offsetted_output = d_output + dstDescPtr->offsetInBytes;
        Rpp16f *temp_output = (Rpp16f *)pDst;
        for (int k = 0; k < dstImgSizes->height; k++)
        //for (int k = 0; k < oSrcSize.height; k++)
        {
            hipMemcpy(temp_output, offsetted_output, elementsInRowout * sizeof (Rpp16f),hipMemcpyDeviceToDevice);
            temp_output += elementsInRowout;
            offsetted_output += elementsInRowMax;
        }

        hipFree(d_input);
        hipFree(d_output);
        hipHostFree(roiTensorPtrSrc);
        hipHostFree(dstImgSizes);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiResize_32f_C1R_Ctx(const Npp32f *pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI, Npp32f *pDst, int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI, int eInterpolation, NppStreamContext nppStreamCtx)
{
        //pln
        int noOfImages = 1;
        int ip_channel = 1;
        RpptDesc srcDesc, dstDesc;
        RpptDescPtr srcDescPtr, dstDescPtr;
        srcDescPtr = &srcDesc;
        dstDescPtr = &dstDesc;

        srcDescPtr->layout = RpptLayout::NCHW;
        dstDescPtr->layout = RpptLayout::NCHW;
        srcDescPtr->dataType = RpptDataType::F32;
        dstDescPtr->dataType = RpptDataType::F32;

        srcDescPtr->numDims = 4;
        dstDescPtr->numDims = 4;

        srcDescPtr->offsetInBytes = 64;
        dstDescPtr->offsetInBytes = 0;

        srcDescPtr->n = noOfImages;
        srcDescPtr->h = oSrcSize.height;
        srcDescPtr->w = oSrcSize.width;
        srcDescPtr->c = ip_channel;

        dstDescPtr->n = noOfImages;
        dstDescPtr->h = oSrcSize.height;
        dstDescPtr->w = oSrcSize.width;
        dstDescPtr->c = ip_channel;

        RppStatus status;
        RpptInterpolationType interpolationType = RpptInterpolationType::BILINEAR;

        if(eInterpolation == NPPI_INTER_UNDEFINED) {
                interpolationType = RpptInterpolationType::NEAREST_NEIGHBOR;
        } else if (eInterpolation == NPPI_INTER_LINEAR) {
                interpolationType = RpptInterpolationType::BILINEAR;
        } else if (eInterpolation == NPPI_INTER_CUBIC) {
                interpolationType = RpptInterpolationType::BICUBIC;
        } else if(eInterpolation == NPPI_INTER_LANCZOS) {
                interpolationType = RpptInterpolationType::LANCZOS;
        } else {
                status = RPP_ERROR_INVALID_ARGUMENTS;
                return(hipRppStatusTocudaNppStatus(status));
        }

    // Optionally set w stride as a multiple of 8 for src/dst

        srcDescPtr->w = ((srcDescPtr->w / 8) * 8) + 8;
        dstDescPtr->w = ((dstDescPtr->w / 8) * 8) + 8;

    // Set n/c/h/w strides for src/dst
        RpptROI *roiTensorPtrSrc;
        hipHostMalloc(&roiTensorPtrSrc, noOfImages * sizeof(RpptROI));
        roiTensorPtrSrc[0].xywhROI.xy.x = 0;
        roiTensorPtrSrc[0].xywhROI.xy.y = 0;
        roiTensorPtrSrc[0].xywhROI.roiWidth = oSrcSize.width;
        roiTensorPtrSrc[0].xywhROI.roiHeight = oSrcSize.height;
        RpptRoiType roiTypeSrc;
        roiTypeSrc = RpptRoiType::XYWH;
        RpptImagePatch *dstImgSizes;
        hipHostMalloc(&dstImgSizes, noOfImages * sizeof(RpptImagePatch));
        dstImgSizes[0].width = oDstSize.width;
        dstImgSizes[0].height = oDstSize.height;

        srcDescPtr->strides.nStride = ip_channel * srcDescPtr->w * srcDescPtr->h;
        srcDescPtr->strides.cStride = srcDescPtr->w * srcDescPtr->h;
        srcDescPtr->strides.hStride = srcDescPtr->w;
        srcDescPtr->strides.wStride = 1;

        dstDescPtr->strides.nStride = ip_channel * dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.cStride = dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.hStride = dstDescPtr->w;
        dstDescPtr->strides.wStride = 1;

        unsigned long long ioBufferSize = 0;
        unsigned long long oBufferSize = 0;

        ioBufferSize = (unsigned long long)srcDescPtr->h * (unsigned long long)srcDescPtr->w * (unsigned long long)srcDescPtr->c * (unsigned long long)noOfImages;
        oBufferSize = (unsigned long long)dstDescPtr->h * (unsigned long long)dstDescPtr->w * (unsigned long long)dstDescPtr->c * (unsigned long long)noOfImages;

        unsigned long long ioBufferSizeInBytes_f32 = (ioBufferSize * 4) + srcDescPtr->offsetInBytes;
	    unsigned long long oBufferSizeInBytes_f32 = (oBufferSize * 4) + dstDescPtr->offsetInBytes;

        Rpp32f *d_input,*d_output;
        hipMalloc(&d_input,ioBufferSizeInBytes_f32);
        hipMalloc(&d_output,oBufferSizeInBytes_f32);
        Rpp32f *offsetted_input, *offsetted_output;
        offsetted_input = d_input + srcDescPtr->offsetInBytes;
        Rpp32f *temp_in = (Rpp32f *)pSrc;
		
        Rpp32u elementsInRow = oSrcSize.width * srcDescPtr->c;

        for (int j = 0; j < oSrcSize.height; j++)
        {
            hipMemcpy(offsetted_input, temp_in, elementsInRow * sizeof (Rpp32f), hipMemcpyDeviceToDevice);
            temp_in += elementsInRow;
            offsetted_input += srcDescPtr->strides.hStride;
        }

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        status = rppt_resize_gpu(d_input, srcDescPtr, d_output, dstDescPtr, dstImgSizes, interpolationType, roiTensorPtrSrc, roiTypeSrc, handle);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        Rpp32u elementsInRowMax = dstDescPtr->w * dstDescPtr->c;
        Rpp32u elementsInRowout = dstImgSizes->width * dstDescPtr->c;
        //Rpp32u elementsInRowout = oSrcSize.width * dstDescPtr->c;
        offsetted_output = d_output + dstDescPtr->offsetInBytes;
        Rpp32f *temp_output = pDst;
        for (int k = 0; k < dstImgSizes->height; k++)
        //for (int k = 0; k < oSrcSize.height; k++)
        {
            hipMemcpy(temp_output, offsetted_output, elementsInRowout * sizeof (Rpp32f),hipMemcpyDeviceToDevice);
            temp_output += elementsInRowout;
            offsetted_output += elementsInRowMax;
        }

        hipFree(d_input);
        hipFree(d_output);
        hipHostFree(roiTensorPtrSrc);
        hipHostFree(dstImgSizes);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiResize_32f_C1R(const Npp32f *pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI, Npp32f *pDst, int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI, int eInterpolation)
{
        //pln
        int noOfImages = 1;
        int ip_channel = 1;
        RpptDesc srcDesc, dstDesc;
        RpptDescPtr srcDescPtr, dstDescPtr;
        srcDescPtr = &srcDesc;
        dstDescPtr = &dstDesc;

        srcDescPtr->layout = RpptLayout::NCHW;
        dstDescPtr->layout = RpptLayout::NCHW;
        srcDescPtr->dataType = RpptDataType::F32;
        dstDescPtr->dataType = RpptDataType::F32;

        srcDescPtr->numDims = 4;
        dstDescPtr->numDims = 4;

        srcDescPtr->offsetInBytes = 64;
        dstDescPtr->offsetInBytes = 0;

        srcDescPtr->n = noOfImages;
        srcDescPtr->h = oSrcSize.height;
        srcDescPtr->w = oSrcSize.width;
        srcDescPtr->c = ip_channel;

        dstDescPtr->n = noOfImages;
        dstDescPtr->h = oSrcSize.height;
        dstDescPtr->w = oSrcSize.width;
        dstDescPtr->c = ip_channel;

        RppStatus status;
        RpptInterpolationType interpolationType = RpptInterpolationType::BILINEAR;

        if(eInterpolation == NPPI_INTER_UNDEFINED) {
                interpolationType = RpptInterpolationType::NEAREST_NEIGHBOR;
        } else if (eInterpolation == NPPI_INTER_LINEAR) {
                interpolationType = RpptInterpolationType::BILINEAR;
        } else if (eInterpolation == NPPI_INTER_CUBIC) {
                interpolationType = RpptInterpolationType::BICUBIC;
        } else if(eInterpolation == NPPI_INTER_LANCZOS) {
                interpolationType = RpptInterpolationType::LANCZOS;
        } else {
                status = RPP_ERROR_INVALID_ARGUMENTS;
                return(hipRppStatusTocudaNppStatus(status));
        }

    // Optionally set w stride as a multiple of 8 for src/dst

        srcDescPtr->w = ((srcDescPtr->w / 8) * 8) + 8;
        dstDescPtr->w = ((dstDescPtr->w / 8) * 8) + 8;

    // Set n/c/h/w strides for src/dst
        RpptROI *roiTensorPtrSrc;
        hipHostMalloc(&roiTensorPtrSrc, noOfImages * sizeof(RpptROI));
        roiTensorPtrSrc[0].xywhROI.xy.x = 0;
        roiTensorPtrSrc[0].xywhROI.xy.y = 0;
        roiTensorPtrSrc[0].xywhROI.roiWidth = oSrcSize.width;
        roiTensorPtrSrc[0].xywhROI.roiHeight = oSrcSize.height;
        RpptRoiType roiTypeSrc;
        roiTypeSrc = RpptRoiType::XYWH;
        RpptImagePatch *dstImgSizes;
        hipHostMalloc(&dstImgSizes, noOfImages * sizeof(RpptImagePatch));
        dstImgSizes[0].width = oDstSize.width;
        dstImgSizes[0].height = oDstSize.height;

        srcDescPtr->strides.nStride = ip_channel * srcDescPtr->w * srcDescPtr->h;
        srcDescPtr->strides.cStride = srcDescPtr->w * srcDescPtr->h;
        srcDescPtr->strides.hStride = srcDescPtr->w;
        srcDescPtr->strides.wStride = 1;

        dstDescPtr->strides.nStride = ip_channel * dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.cStride = dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.hStride = dstDescPtr->w;
        dstDescPtr->strides.wStride = 1;

        unsigned long long ioBufferSize = 0;
        unsigned long long oBufferSize = 0;

        ioBufferSize = (unsigned long long)srcDescPtr->h * (unsigned long long)srcDescPtr->w * (unsigned long long)srcDescPtr->c * (unsigned long long)noOfImages;
        oBufferSize = (unsigned long long)dstDescPtr->h * (unsigned long long)dstDescPtr->w * (unsigned long long)dstDescPtr->c * (unsigned long long)noOfImages;

        unsigned long long ioBufferSizeInBytes_f32 = (ioBufferSize * 4) + srcDescPtr->offsetInBytes;
	    unsigned long long oBufferSizeInBytes_f32 = (oBufferSize * 4) + dstDescPtr->offsetInBytes;

        Rpp32f *d_input,*d_output;
        hipMalloc(&d_input,ioBufferSizeInBytes_f32);
        hipMalloc(&d_output,oBufferSizeInBytes_f32);
        Rpp32f *offsetted_input, *offsetted_output;
        offsetted_input = d_input + srcDescPtr->offsetInBytes;
        Rpp32f *temp_in = (Rpp32f *)pSrc;
		
        Rpp32u elementsInRow = oSrcSize.width * srcDescPtr->c;

        for (int j = 0; j < oSrcSize.height; j++)
        {
            hipMemcpy(offsetted_input, temp_in, elementsInRow * sizeof (Rpp32f), hipMemcpyDeviceToDevice);
            temp_in += elementsInRow;
            offsetted_input += srcDescPtr->strides.hStride;
        }

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        status = rppt_resize_gpu(d_input, srcDescPtr, d_output, dstDescPtr, dstImgSizes, interpolationType, roiTensorPtrSrc, roiTypeSrc, handle);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        Rpp32u elementsInRowMax = dstDescPtr->w * dstDescPtr->c;
        Rpp32u elementsInRowout = dstImgSizes->width * dstDescPtr->c;
        //Rpp32u elementsInRowout = oSrcSize.width * dstDescPtr->c;
        offsetted_output = d_output + dstDescPtr->offsetInBytes;
        Rpp32f *temp_output = pDst;
        for (int k = 0; k < dstImgSizes->height; k++)
        //for (int k = 0; k < oSrcSize.height; k++)
        {
            hipMemcpy(temp_output, offsetted_output, elementsInRowout * sizeof (Rpp32f),hipMemcpyDeviceToDevice);
            temp_output += elementsInRowout;
            offsetted_output += elementsInRowMax;
        }

        hipFree(d_input);
        hipFree(d_output);
        hipHostFree(roiTensorPtrSrc);
        hipHostFree(dstImgSizes);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiResize_32f_C3R_Ctx(const Npp32f *pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI, Npp32f *pDst, int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI, int eInterpolation, NppStreamContext nppStreamCtx)
{
        //pln
        int noOfImages = 1;
        int ip_channel = 3;
        RpptDesc srcDesc, dstDesc;
        RpptDescPtr srcDescPtr, dstDescPtr;
        srcDescPtr = &srcDesc;
        dstDescPtr = &dstDesc;

        srcDescPtr->layout = RpptLayout::NHWC;
        dstDescPtr->layout = RpptLayout::NHWC;
        srcDescPtr->dataType = RpptDataType::F32;
        dstDescPtr->dataType = RpptDataType::F32;

        srcDescPtr->numDims = 4;
        dstDescPtr->numDims = 4;

        srcDescPtr->offsetInBytes = 64;
        dstDescPtr->offsetInBytes = 0;

        srcDescPtr->n = noOfImages;
        srcDescPtr->h = oSrcSize.height;
        srcDescPtr->w = oSrcSize.width;
        srcDescPtr->c = ip_channel;

        dstDescPtr->n = noOfImages;
        dstDescPtr->h = oSrcSize.height;
        dstDescPtr->w = oSrcSize.width;
        dstDescPtr->c = ip_channel;

        RppStatus status;
        RpptInterpolationType interpolationType = RpptInterpolationType::BILINEAR;

        if(eInterpolation == NPPI_INTER_UNDEFINED) {
                interpolationType = RpptInterpolationType::NEAREST_NEIGHBOR;
        } else if (eInterpolation == NPPI_INTER_LINEAR) {
                interpolationType = RpptInterpolationType::BILINEAR;
        } else if (eInterpolation == NPPI_INTER_CUBIC) {
                interpolationType = RpptInterpolationType::BICUBIC;
        } else if(eInterpolation == NPPI_INTER_LANCZOS) {
                interpolationType = RpptInterpolationType::LANCZOS;
        } else {
                status = RPP_ERROR_INVALID_ARGUMENTS;
                return(hipRppStatusTocudaNppStatus(status));
        }

    // Optionally set w stride as a multiple of 8 for src/dst

        srcDescPtr->w = ((srcDescPtr->w / 8) * 8) + 8;
        dstDescPtr->w = ((dstDescPtr->w / 8) * 8) + 8;

    // Set n/c/h/w strides for src/dst
        RpptROI *roiTensorPtrSrc;
        hipHostMalloc(&roiTensorPtrSrc, noOfImages * sizeof(RpptROI));
        roiTensorPtrSrc[0].xywhROI.xy.x = 0;
        roiTensorPtrSrc[0].xywhROI.xy.y = 0;
        roiTensorPtrSrc[0].xywhROI.roiWidth = oSrcSize.width;
        roiTensorPtrSrc[0].xywhROI.roiHeight = oSrcSize.height;
        RpptRoiType roiTypeSrc;
        roiTypeSrc = RpptRoiType::XYWH;
        RpptImagePatch *dstImgSizes;
        hipHostMalloc(&dstImgSizes, noOfImages * sizeof(RpptImagePatch));
        dstImgSizes[0].width = oDstSize.width;
        dstImgSizes[0].height = oDstSize.height;

        srcDescPtr->strides.nStride = srcDescPtr->c * srcDescPtr->w * srcDescPtr->h;
        srcDescPtr->strides.hStride = srcDescPtr->c * srcDescPtr->w;
        srcDescPtr->strides.wStride = srcDescPtr->c;
        srcDescPtr->strides.cStride = 1;

        dstDescPtr->strides.nStride = dstDescPtr->c * dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.hStride = dstDescPtr->c * dstDescPtr->w;
        dstDescPtr->strides.wStride = dstDescPtr->c;
        dstDescPtr->strides.cStride = 1;

        unsigned long long ioBufferSize = 0;
        unsigned long long oBufferSize = 0;

        ioBufferSize = (unsigned long long)srcDescPtr->h * (unsigned long long)srcDescPtr->w * (unsigned long long)srcDescPtr->c * (unsigned long long)noOfImages;
        oBufferSize = (unsigned long long)dstDescPtr->h * (unsigned long long)dstDescPtr->w * (unsigned long long)dstDescPtr->c * (unsigned long long)noOfImages;

        unsigned long long ioBufferSizeInBytes_f32 = (ioBufferSize * 4) + srcDescPtr->offsetInBytes;
	    unsigned long long oBufferSizeInBytes_f32 = (oBufferSize * 4) + dstDescPtr->offsetInBytes;

        Rpp32f *d_input,*d_output;
        hipMalloc(&d_input,ioBufferSizeInBytes_f32);
        hipMalloc(&d_output,oBufferSizeInBytes_f32);
        Rpp32f *offsetted_input, *offsetted_output;
        offsetted_input = d_input + srcDescPtr->offsetInBytes;
        Rpp32f *temp_in = (Rpp32f *)pSrc;
		
        Rpp32u elementsInRow = oSrcSize.width * srcDescPtr->c;

        for (int j = 0; j < oSrcSize.height; j++)
        {
            hipMemcpy(offsetted_input, temp_in, elementsInRow * sizeof (Rpp32f), hipMemcpyDeviceToDevice);
            temp_in += elementsInRow;
            offsetted_input += srcDescPtr->strides.hStride;
        }

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        status = rppt_resize_gpu(d_input, srcDescPtr, d_output, dstDescPtr, dstImgSizes, interpolationType, roiTensorPtrSrc, roiTypeSrc, handle);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        Rpp32u elementsInRowMax = dstDescPtr->w * dstDescPtr->c;
        Rpp32u elementsInRowout = dstImgSizes->width * dstDescPtr->c;
        //Rpp32u elementsInRowout = oSrcSize.width * dstDescPtr->c;
        offsetted_output = d_output + dstDescPtr->offsetInBytes;
        Rpp32f *temp_output = pDst;
        for (int k = 0; k < dstImgSizes->height; k++)
        //for (int k = 0; k < oSrcSize.height; k++)
        {
            hipMemcpy(temp_output, offsetted_output, elementsInRowout * sizeof (Rpp32f),hipMemcpyDeviceToDevice);
            temp_output += elementsInRowout;
            offsetted_output += elementsInRowMax;
        }

        hipFree(d_input);
        hipFree(d_output);
        hipHostFree(roiTensorPtrSrc);
        hipHostFree(dstImgSizes);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiResize_32f_C3R(const Npp32f *pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI, Npp32f *pDst, int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI, int eInterpolation)
{
        //pln
        int noOfImages = 1;
        int ip_channel = 3;
        RpptDesc srcDesc, dstDesc;
        RpptDescPtr srcDescPtr, dstDescPtr;
        srcDescPtr = &srcDesc;
        dstDescPtr = &dstDesc;

        srcDescPtr->layout = RpptLayout::NHWC;
        dstDescPtr->layout = RpptLayout::NHWC;
        srcDescPtr->dataType = RpptDataType::F32;
        dstDescPtr->dataType = RpptDataType::F32;

        srcDescPtr->numDims = 4;
        dstDescPtr->numDims = 4;

        srcDescPtr->offsetInBytes = 64;
        dstDescPtr->offsetInBytes = 0;

        srcDescPtr->n = noOfImages;
        srcDescPtr->h = oSrcSize.height;
        srcDescPtr->w = oSrcSize.width;
        srcDescPtr->c = ip_channel;

        dstDescPtr->n = noOfImages;
        dstDescPtr->h = oSrcSize.height;
        dstDescPtr->w = oSrcSize.width;
        dstDescPtr->c = ip_channel;

        RppStatus status;
        RpptInterpolationType interpolationType = RpptInterpolationType::BILINEAR;

        if(eInterpolation == NPPI_INTER_UNDEFINED) {
                interpolationType = RpptInterpolationType::NEAREST_NEIGHBOR;
        } else if (eInterpolation == NPPI_INTER_LINEAR) {
                interpolationType = RpptInterpolationType::BILINEAR;
        } else if (eInterpolation == NPPI_INTER_CUBIC) {
                interpolationType = RpptInterpolationType::BICUBIC;
        } else if(eInterpolation == NPPI_INTER_LANCZOS) {
                interpolationType = RpptInterpolationType::LANCZOS;
        } else {
                status = RPP_ERROR_INVALID_ARGUMENTS;
                return(hipRppStatusTocudaNppStatus(status));
        }

    // Optionally set w stride as a multiple of 8 for src/dst

        srcDescPtr->w = ((srcDescPtr->w / 8) * 8) + 8;
        dstDescPtr->w = ((dstDescPtr->w / 8) * 8) + 8;

    // Set n/c/h/w strides for src/dst
        RpptROI *roiTensorPtrSrc;
        hipHostMalloc(&roiTensorPtrSrc, noOfImages * sizeof(RpptROI));
        roiTensorPtrSrc[0].xywhROI.xy.x = 0;
        roiTensorPtrSrc[0].xywhROI.xy.y = 0;
        roiTensorPtrSrc[0].xywhROI.roiWidth = oSrcSize.width;
        roiTensorPtrSrc[0].xywhROI.roiHeight = oSrcSize.height;
        RpptRoiType roiTypeSrc;
        roiTypeSrc = RpptRoiType::XYWH;
        RpptImagePatch *dstImgSizes;
        hipHostMalloc(&dstImgSizes, noOfImages * sizeof(RpptImagePatch));
        dstImgSizes[0].width = oDstSize.width;
        dstImgSizes[0].height = oDstSize.height;

        srcDescPtr->strides.nStride = srcDescPtr->c * srcDescPtr->w * srcDescPtr->h;
        srcDescPtr->strides.hStride = srcDescPtr->c * srcDescPtr->w;
        srcDescPtr->strides.wStride = srcDescPtr->c;
        srcDescPtr->strides.cStride = 1;

        dstDescPtr->strides.nStride = dstDescPtr->c * dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.hStride = dstDescPtr->c * dstDescPtr->w;
        dstDescPtr->strides.wStride = dstDescPtr->c;
        dstDescPtr->strides.cStride = 1;

        unsigned long long ioBufferSize = 0;
        unsigned long long oBufferSize = 0;

        ioBufferSize = (unsigned long long)srcDescPtr->h * (unsigned long long)srcDescPtr->w * (unsigned long long)srcDescPtr->c * (unsigned long long)noOfImages;
        oBufferSize = (unsigned long long)dstDescPtr->h * (unsigned long long)dstDescPtr->w * (unsigned long long)dstDescPtr->c * (unsigned long long)noOfImages;

        unsigned long long ioBufferSizeInBytes_f32 = (ioBufferSize * 4) + srcDescPtr->offsetInBytes;
	    unsigned long long oBufferSizeInBytes_f32 = (oBufferSize * 4) + dstDescPtr->offsetInBytes;

        Rpp32f *d_input,*d_output;
        hipMalloc(&d_input,ioBufferSizeInBytes_f32);
        hipMalloc(&d_output,oBufferSizeInBytes_f32);
        Rpp32f *offsetted_input, *offsetted_output;
        offsetted_input = d_input + srcDescPtr->offsetInBytes;
        Rpp32f *temp_in = (Rpp32f *)pSrc;
		
        Rpp32u elementsInRow = oSrcSize.width * srcDescPtr->c;

        for (int j = 0; j < oSrcSize.height; j++)
        {
            hipMemcpy(offsetted_input, temp_in, elementsInRow * sizeof (Rpp32f), hipMemcpyDeviceToDevice);
            temp_in += elementsInRow;
            offsetted_input += srcDescPtr->strides.hStride;
        }

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        status = rppt_resize_gpu(d_input, srcDescPtr, d_output, dstDescPtr, dstImgSizes, interpolationType, roiTensorPtrSrc, roiTypeSrc, handle);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        Rpp32u elementsInRowMax = dstDescPtr->w * dstDescPtr->c;
        Rpp32u elementsInRowout = dstImgSizes->width * dstDescPtr->c;
        //Rpp32u elementsInRowout = oSrcSize.width * dstDescPtr->c;
        offsetted_output = d_output + dstDescPtr->offsetInBytes;
        Rpp32f *temp_output = pDst;
        for (int k = 0; k < dstImgSizes->height; k++)
        //for (int k = 0; k < oSrcSize.height; k++)
        {
            hipMemcpy(temp_output, offsetted_output, elementsInRowout * sizeof (Rpp32f),hipMemcpyDeviceToDevice);
            temp_output += elementsInRowout;
            offsetted_output += elementsInRowMax;
        }

        hipFree(d_input);
        hipFree(d_output);
        hipHostFree(roiTensorPtrSrc);
        hipHostFree(dstImgSizes);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiResize_32f_P3R_Ctx(const Npp32f *pSrc[3], int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI, Npp32f *pDst[3], int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI, int eInterpolation, NppStreamContext nppStreamCtx)
{
        //pln3
        int noOfImages = 1;
        int ip_channel = 3;
        RpptDesc srcDesc, dstDesc;
        RpptDescPtr srcDescPtr, dstDescPtr;
        srcDescPtr = &srcDesc;
        dstDescPtr = &dstDesc;

        srcDescPtr->layout = RpptLayout::NCHW;
        dstDescPtr->layout = RpptLayout::NCHW;
        srcDescPtr->dataType = RpptDataType::F32;
        dstDescPtr->dataType = RpptDataType::F32;

        srcDescPtr->numDims = 4;
        dstDescPtr->numDims = 4;

        srcDescPtr->offsetInBytes = 64;
        dstDescPtr->offsetInBytes = 0;

        srcDescPtr->n = noOfImages;
        srcDescPtr->h = oSrcSize.height;
        srcDescPtr->w = oSrcSize.width;
        srcDescPtr->c = ip_channel;

        dstDescPtr->n = noOfImages;
        dstDescPtr->h = oDstSize.height;
        dstDescPtr->w = oDstSize.width;
        dstDescPtr->c = ip_channel;

        RppStatus status;
        RpptInterpolationType interpolationType = RpptInterpolationType::BILINEAR;

        if(eInterpolation == NPPI_INTER_UNDEFINED) {
                interpolationType = RpptInterpolationType::NEAREST_NEIGHBOR;
        } else if (eInterpolation == NPPI_INTER_LINEAR) {
                interpolationType = RpptInterpolationType::BILINEAR;
        } else if (eInterpolation == NPPI_INTER_CUBIC) {
                interpolationType = RpptInterpolationType::BICUBIC;
        } else if(eInterpolation == NPPI_INTER_LANCZOS) {
                interpolationType = RpptInterpolationType::LANCZOS;
        } else {
                status = RPP_ERROR_INVALID_ARGUMENTS;
                return(hipRppStatusTocudaNppStatus(status));
        }

    // Optionally set w stride as a multiple of 8 for src/dst

        srcDescPtr->w = ((srcDescPtr->w / 8) * 8) + 8;
        dstDescPtr->w = ((dstDescPtr->w / 8) * 8) + 8;

    // Set n/c/h/w strides for src/dst
        RpptROI *roiTensorPtrSrc;
        hipHostMalloc(&roiTensorPtrSrc, noOfImages * sizeof(RpptROI));
        roiTensorPtrSrc[0].xywhROI.xy.x = 0;
        roiTensorPtrSrc[0].xywhROI.xy.y = 0;
        roiTensorPtrSrc[0].xywhROI.roiWidth = oSrcSize.width;
        roiTensorPtrSrc[0].xywhROI.roiHeight = oSrcSize.height;
        RpptRoiType roiTypeSrc;
        roiTypeSrc = RpptRoiType::XYWH;
        RpptImagePatch *dstImgSizes;
        hipHostMalloc(&dstImgSizes, noOfImages * sizeof(RpptImagePatch));
        dstImgSizes[0].width = oDstSize.width;
        dstImgSizes[0].height = oDstSize.height;

        srcDescPtr->strides.nStride = srcDescPtr->c * srcDescPtr->w * srcDescPtr->h;
        srcDescPtr->strides.cStride = srcDescPtr->w * srcDescPtr->h;
        srcDescPtr->strides.hStride = srcDescPtr->w;
        srcDescPtr->strides.wStride = 1;

        dstDescPtr->strides.nStride = dstDescPtr->c * dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.cStride = dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.hStride = dstDescPtr->w;
        dstDescPtr->strides.wStride = 1;

        unsigned long long ioBufferSize = 0;
        unsigned long long oBufferSize = 0;

        ioBufferSize = (unsigned long long)srcDescPtr->h * (unsigned long long)srcDescPtr->w * (unsigned long long)srcDescPtr->c * (unsigned long long)noOfImages;
        oBufferSize = (unsigned long long)dstDescPtr->h * (unsigned long long)dstDescPtr->w * (unsigned long long)dstDescPtr->c * (unsigned long long)noOfImages;

        unsigned long long ioBufferSizeInBytes_f32 = (ioBufferSize * 4) + srcDescPtr->offsetInBytes;
        unsigned long long oBufferSizeInBytes_f32 = (oBufferSize * 4)+ dstDescPtr->offsetInBytes;

        Rpp32f *d_input,*d_output;
        hipMalloc(&d_input,ioBufferSizeInBytes_f32);
        hipMalloc(&d_output,oBufferSizeInBytes_f32);
        Rpp32f *offsetted_input, *offsetted_output;
        offsetted_input = d_input + srcDescPtr->offsetInBytes;
        
        Rpp32f *temp_in;
        hipMalloc(&temp_in, oSrcSize.height * oSrcSize.width * sizeof(Rpp32f) * ip_channel);
        for (int i = 0; i < ip_channel; i++)
	    {
            hipMemcpy(temp_in + (i * oSrcSize.height * oSrcSize.width), pSrc[i], oSrcSize.height * oSrcSize.width, hipMemcpyDeviceToDevice);
        }

        Rpp32u elementsInRow = oSrcSize.width * srcDescPtr->c;

        for (int j = 0; j < oSrcSize.height; j++)
        {
            hipMemcpy(offsetted_input, temp_in, elementsInRow * sizeof (Rpp32f), hipMemcpyDeviceToDevice);
            temp_in += elementsInRow;
            offsetted_input += srcDescPtr->w * srcDescPtr->c;
        }

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        status = rppt_resize_gpu((RppPtr_t)d_input, srcDescPtr, (RppPtr_t)d_output, dstDescPtr, dstImgSizes, interpolationType, roiTensorPtrSrc, roiTypeSrc, handle);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        Rpp32u elementsInRowMax = dstDescPtr->w * dstDescPtr->c;
        Rpp32u elementsInRowout = dstImgSizes->width * dstDescPtr->c;
        //Rpp32u elementsInRowout = oSrcSize.width * dstDescPtr->c;
        offsetted_output = d_output + dstDescPtr->offsetInBytes;
        Rpp32f *temp_output;
        hipMalloc(&temp_output, oDstSize.height * oDstSize.width * ip_channel);
        for (int k = 0; k < oDstSize.height; k++)
        {
            hipMemcpy(temp_output + (k * elementsInRowout), offsetted_output, elementsInRowout * sizeof (Rpp32f),hipMemcpyDeviceToDevice);
            //temp_output += elementsInRowout;
            offsetted_output += elementsInRowMax;
        }
        for (int m = 0; m < ip_channel; m++)
        {
            hipMemcpy(pDst[m],temp_output + (m * oDstSize.height * oDstSize.width), oDstSize.height * oDstSize.width, hipMemcpyDeviceToDevice);
        }


        hipFree(d_input);
        hipFree(d_output);
        hipHostFree(roiTensorPtrSrc);
        hipHostFree(dstImgSizes);
        hipFree(temp_in);
        hipFree(temp_output);

        return(hipRppStatusTocudaNppStatus(status));
}
		
NppStatus nppiResize_32f_P3R(const Npp32f *pSrc[3], int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI, Npp32f *pDst[3], int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI, int eInterpolation)
{
        //pln3
        int noOfImages = 1;
        int ip_channel = 3;
        RpptDesc srcDesc, dstDesc;
        RpptDescPtr srcDescPtr, dstDescPtr;
        srcDescPtr = &srcDesc;
        dstDescPtr = &dstDesc;

        srcDescPtr->layout = RpptLayout::NCHW;
        dstDescPtr->layout = RpptLayout::NCHW;
        srcDescPtr->dataType = RpptDataType::F32;
        dstDescPtr->dataType = RpptDataType::F32;

        srcDescPtr->numDims = 4;
        dstDescPtr->numDims = 4;

        srcDescPtr->offsetInBytes = 64;
        dstDescPtr->offsetInBytes = 0;

        srcDescPtr->n = noOfImages;
        srcDescPtr->h = oSrcSize.height;
        srcDescPtr->w = oSrcSize.width;
        srcDescPtr->c = ip_channel;

        dstDescPtr->n = noOfImages;
        dstDescPtr->h = oDstSize.height;
        dstDescPtr->w = oDstSize.width;
        dstDescPtr->c = ip_channel;

        RppStatus status;
        RpptInterpolationType interpolationType = RpptInterpolationType::BILINEAR;

        if(eInterpolation == NPPI_INTER_UNDEFINED) {
                interpolationType = RpptInterpolationType::NEAREST_NEIGHBOR;
        } else if (eInterpolation == NPPI_INTER_LINEAR) {
                interpolationType = RpptInterpolationType::BILINEAR;
        } else if (eInterpolation == NPPI_INTER_CUBIC) {
                interpolationType = RpptInterpolationType::BICUBIC;
        } else if(eInterpolation == NPPI_INTER_LANCZOS) {
                interpolationType = RpptInterpolationType::LANCZOS;
        } else {
                status = RPP_ERROR_INVALID_ARGUMENTS;
                return(hipRppStatusTocudaNppStatus(status));
        }

    // Optionally set w stride as a multiple of 8 for src/dst

        srcDescPtr->w = ((srcDescPtr->w / 8) * 8) + 8;
        dstDescPtr->w = ((dstDescPtr->w / 8) * 8) + 8;

    // Set n/c/h/w strides for src/dst
        RpptROI *roiTensorPtrSrc;
        hipHostMalloc(&roiTensorPtrSrc, noOfImages * sizeof(RpptROI));
        roiTensorPtrSrc[0].xywhROI.xy.x = 0;
        roiTensorPtrSrc[0].xywhROI.xy.y = 0;
        roiTensorPtrSrc[0].xywhROI.roiWidth = oSrcSize.width;
        roiTensorPtrSrc[0].xywhROI.roiHeight = oSrcSize.height;
        RpptRoiType roiTypeSrc;
        roiTypeSrc = RpptRoiType::XYWH;
        RpptImagePatch *dstImgSizes;
        hipHostMalloc(&dstImgSizes, noOfImages * sizeof(RpptImagePatch));
        dstImgSizes[0].width = oDstSize.width;
        dstImgSizes[0].height = oDstSize.height;

        srcDescPtr->strides.nStride = srcDescPtr->c * srcDescPtr->w * srcDescPtr->h;
        srcDescPtr->strides.cStride = srcDescPtr->w * srcDescPtr->h;
        srcDescPtr->strides.hStride = srcDescPtr->w;
        srcDescPtr->strides.wStride = 1;

        dstDescPtr->strides.nStride = dstDescPtr->c * dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.cStride = dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.hStride = dstDescPtr->w;
        dstDescPtr->strides.wStride = 1;

        unsigned long long ioBufferSize = 0;
        unsigned long long oBufferSize = 0;

        ioBufferSize = (unsigned long long)srcDescPtr->h * (unsigned long long)srcDescPtr->w * (unsigned long long)srcDescPtr->c * (unsigned long long)noOfImages;
        oBufferSize = (unsigned long long)dstDescPtr->h * (unsigned long long)dstDescPtr->w * (unsigned long long)dstDescPtr->c * (unsigned long long)noOfImages;

        unsigned long long ioBufferSizeInBytes_f32 = (ioBufferSize * 4) + srcDescPtr->offsetInBytes;
        unsigned long long oBufferSizeInBytes_f32 = (oBufferSize * 4)+ dstDescPtr->offsetInBytes;

        Rpp32f *d_input,*d_output;
        hipMalloc(&d_input,ioBufferSizeInBytes_f32);
        hipMalloc(&d_output,oBufferSizeInBytes_f32);
        Rpp32f *offsetted_input, *offsetted_output;
        offsetted_input = d_input + srcDescPtr->offsetInBytes;
        
        Rpp32f *temp_in;
        hipMalloc(&temp_in, oSrcSize.height * oSrcSize.width * sizeof(Rpp32f) * ip_channel);
        for (int i = 0; i < ip_channel; i++)
	    {
            hipMemcpy(temp_in + (i * oSrcSize.height * oSrcSize.width), pSrc[i], oSrcSize.height * oSrcSize.width, hipMemcpyDeviceToDevice);
        }

        Rpp32u elementsInRow = oSrcSize.width * srcDescPtr->c;

        for (int j = 0; j < oSrcSize.height; j++)
        {
            hipMemcpy(offsetted_input, temp_in, elementsInRow * sizeof (Rpp32f), hipMemcpyDeviceToDevice);
            temp_in += elementsInRow;
            offsetted_input += srcDescPtr->w * srcDescPtr->c;
        }

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        status = rppt_resize_gpu((RppPtr_t)d_input, srcDescPtr, (RppPtr_t)d_output, dstDescPtr, dstImgSizes, interpolationType, roiTensorPtrSrc, roiTypeSrc, handle);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        Rpp32u elementsInRowMax = dstDescPtr->w * dstDescPtr->c;
        Rpp32u elementsInRowout = dstImgSizes->width * dstDescPtr->c;
        //Rpp32u elementsInRowout = oSrcSize.width * dstDescPtr->c;
        offsetted_output = d_output + dstDescPtr->offsetInBytes;
        Rpp32f *temp_output;
        hipMalloc(&temp_output, oDstSize.height * oDstSize.width * ip_channel);
        for (int k = 0; k < oDstSize.height; k++)
        {
            hipMemcpy(temp_output + (k * elementsInRowout), offsetted_output, elementsInRowout * sizeof (Rpp32f),hipMemcpyDeviceToDevice);
            //temp_output += elementsInRowout;
            offsetted_output += elementsInRowMax;
        }
        for (int m = 0; m < ip_channel; m++)
        {
            hipMemcpy(pDst[m],temp_output + (m * oDstSize.height * oDstSize.width), oDstSize.height * oDstSize.width, hipMemcpyDeviceToDevice);
        }


        hipFree(d_input);
        hipFree(d_output);
        hipHostFree(roiTensorPtrSrc);
        hipHostFree(dstImgSizes);
        hipFree(temp_in);
        hipFree(temp_output);

        return(hipRppStatusTocudaNppStatus(status));
}

/*NppStatus nppiResize_8u_P3R_Ctx(const Npp8u *pSrc[3], int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI, Npp8u *pDst[3], int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI, int eInterpolation, NppStreamContext nppStreamCtx)
{
        //pln3
        int noOfImages = 1;
        int ip_channel = 3;
        RpptDesc srcDesc, dstDesc;
        RpptDescPtr srcDescPtr, dstDescPtr;
        srcDescPtr = &srcDesc;
        dstDescPtr = &dstDesc;

        srcDescPtr->layout = RpptLayout::NCHW;
        dstDescPtr->layout = RpptLayout::NCHW;
        srcDescPtr->dataType = RpptDataType::U8;
        dstDescPtr->dataType = RpptDataType::U8;

        srcDescPtr->numDims = 4;
        dstDescPtr->numDims = 4;

        srcDescPtr->offsetInBytes = 64;
        dstDescPtr->offsetInBytes = 0;

        srcDescPtr->n = noOfImages;
        srcDescPtr->h = oSrcSize.height;
        srcDescPtr->w = oSrcSize.width;
        srcDescPtr->c = ip_channel;

        dstDescPtr->n = noOfImages;
        dstDescPtr->h = oDstSize.height;
        dstDescPtr->w = oDstSize.width;
        dstDescPtr->c = ip_channel;

    // Optionally set w stride as a multiple of 8 for src/dst

        srcDescPtr->w = ((srcDescPtr->w / 8) * 8) + 8;
        dstDescPtr->w = ((dstDescPtr->w / 8) * 8) + 8;

    // Set n/c/h/w strides for src/dst
        RpptROI *roiTensorPtrSrc;
        hipHostMalloc(&roiTensorPtrSrc, noOfImages * sizeof(RpptROI));
        roiTensorPtrSrc[0].xywhROI.xy.x = 0;
        roiTensorPtrSrc[0].xywhROI.xy.y = 0;
        roiTensorPtrSrc[0].xywhROI.roiWidth = oSrcSize.width;
        roiTensorPtrSrc[0].xywhROI.roiHeight = oSrcSize.height;
        RpptRoiType roiTypeSrc;
        roiTypeSrc = RpptRoiType::XYWH;
        RpptImagePatch *dstImgSizes;
        hipHostMalloc(&dstImgSizes, noOfImages * sizeof(RpptImagePatch));
        dstImgSizes[0].width = oDstSize.width;
        dstImgSizes[0].height = oDstSize.height;

        srcDescPtr->strides.nStride = srcDescPtr->c * srcDescPtr->w * srcDescPtr->h;
	srcDescPtr->strides.cStride = srcDescPtr->w * srcDescPtr->h;
	srcDescPtr->strides.hStride = srcDescPtr->w;
	srcDescPtr->strides.wStride = 1;

        dstDescPtr->strides.nStride = dstDescPtr->c * dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.cStride = dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.hStride = dstDescPtr->w;
        dstDescPtr->strides.wStride = 1;

        unsigned long long ioBufferSize = 0;
        unsigned long long oBufferSize = 0;

        ioBufferSize = (unsigned long long)srcDescPtr->h * (unsigned long long)srcDescPtr->w * (unsigned long long)srcDescPtr->c * (unsigned long long)noOfImages;
        oBufferSize = (unsigned long long)dstDescPtr->h * (unsigned long long)dstDescPtr->w * (unsigned long long)dstDescPtr->c * (unsigned long long)noOfImages;

        unsigned long long ioBufferSizeInBytes_u8 = ioBufferSize + srcDescPtr->offsetInBytes;
        unsigned long long oBufferSizeInBytes_u8 = oBufferSize + dstDescPtr->offsetInBytes;

        Rpp8u *d_input,*d_output;
        hipMalloc(&d_input,ioBufferSizeInBytes_u8);
        hipMalloc(&d_output,oBufferSizeInBytes_u8);
	Rpp8u *offsetted_input, *offsetted_output;
        offsetted_input = d_input + srcDescPtr->offsetInBytes;
        
	Rpp8u *temp_in;
        hipMalloc(&temp_in, oSrcSize.height * oSrcSize.width * ip_channel);
        for (int i = 0; i < ip_channel; i++)
        {
            hipMemcpy(temp_in + (i * oSrcSize.height * oSrcSize.width), pSrc[i], oSrcSize.height * oSrcSize.width, hipMemcpyDeviceToDevice);
        }

        Rpp32u elementsInRow = oSrcSize.width * srcDescPtr->c;

        for (int j = 0; j < oSrcSize.height; j++)
        {
            hipMemcpy(offsetted_input, temp_in, elementsInRow * sizeof (Rpp8u), hipMemcpyDeviceToDevice);
            temp_in += elementsInRow;
            offsetted_input += srcDescPtr->w * srcDescPtr->c;
        }


        RppStatus status;
        RpptInterpolationType interpolationType = RpptInterpolationType::BILINEAR;

        if(eInterpolation == NPPI_INTER_UNDEFINED) {
                interpolationType = RpptInterpolationType::NEAREST_NEIGHBOR;
        } else if (eInterpolation == NPPI_INTER_LINEAR) {
                interpolationType = RpptInterpolationType::BILINEAR;
        } else if (eInterpolation == NPPI_INTER_CUBIC) {
                interpolationType = RpptInterpolationType::BICUBIC;
        } else if(eInterpolation == NPPI_INTER_LANCZOS) {
                interpolationType = RpptInterpolationType::LANCZOS;
        } else {
                status = RPP_ERROR_INVALID_ARGUMENTS;
                return(hipRppStatusTocudaNppStatus(status));
        }

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);
        clock_t start, end;
        double gpu_time_used;

        start = clock();
        status = rppt_resize_gpu((RppPtr_t)d_input, srcDescPtr, (RppPtr_t)d_output, dstDescPtr, dstImgSizes, interpolationType, roiTensorPtrSrc, roiTypeSrc, handle);
        hipDeviceSynchronize();
        end = clock();

        printf("\nnppiResize_8u_P3R_Ctx is %d\n", status);
        gpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
        std::cout << "\nGPU Time - Resize_8u : " << gpu_time_used;
        printf("\n");

        rppDestroyGPU(handle);
        Rpp32u elementsInRowMax = dstDescPtr->w * dstDescPtr->c;
        Rpp32u elementsInRowout = dstImgSizes->width * dstDescPtr->c;
        //Rpp32u elementsInRowout = oSrcSize.width * dstDescPtr->c;
        offsetted_output = d_output + dstDescPtr->offsetInBytes;
        Rpp8u *temp_output;
        hipMalloc(&temp_output, oDstSize.height * oDstSize.width * ip_channel);
        for (int k = 0; k < oDstSize.height; k++)
        {
            hipMemcpy(temp_output + (k * elementsInRowout), offsetted_output, elementsInRowout * sizeof (Rpp8u),hipMemcpyDeviceToDevice);
            //temp_output += elementsInRowout;
            offsetted_output += elementsInRowMax;
        }
        for (int m = 0; m < ip_channel; m++)
        {
            hipMemcpy(pDst[m],temp_output + (m * oDstSize.height * oDstSize.width), oDstSize.height * oDstSize.width, hipMemcpyDeviceToDevice);
        }


        hipFree(d_input);
        hipFree(d_output);
        hipHostFree(roiTensorPtrSrc);
        hipHostFree(dstImgSizes);
	hipFree(temp_in);
        hipFree(temp_output);

        return(hipRppStatusTocudaNppStatus(status));
}*/

/*NppStatus nppiResize_8u_P3R(const Npp8u *pSrc[3], int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI, Npp8u *pDst[3], int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI, int eInterpolation)
{
        //pln3
        int noOfImages = 1;
        int ip_channel = 3;
        RpptDesc srcDesc, dstDesc;
        RpptDescPtr srcDescPtr, dstDescPtr;
        srcDescPtr = &srcDesc;
        dstDescPtr = &dstDesc;

        srcDescPtr->layout = RpptLayout::NCHW;
        dstDescPtr->layout = RpptLayout::NCHW;
        srcDescPtr->dataType = RpptDataType::U8;
        dstDescPtr->dataType = RpptDataType::U8;

        srcDescPtr->numDims = 4;
        dstDescPtr->numDims = 4;

        srcDescPtr->offsetInBytes = 64;
        dstDescPtr->offsetInBytes = 0;

        srcDescPtr->n = noOfImages;
        srcDescPtr->h = oSrcSize.height;
        srcDescPtr->w = oSrcSize.width;
        srcDescPtr->c = ip_channel;

        dstDescPtr->n = noOfImages;
        dstDescPtr->h = oSrcSize.height;
        dstDescPtr->w = oSrcSize.width;
        dstDescPtr->c = ip_channel;

    // Optionally set w stride as a multiple of 8 for src/dst

        srcDescPtr->w = ((srcDescPtr->w / 8) * 8) + 8;
        dstDescPtr->w = ((dstDescPtr->w / 8) * 8) + 8;

    // Set n/c/h/w strides for src/dst
        RpptROI *roiTensorPtrSrc;
        hipHostMalloc(&roiTensorPtrSrc, noOfImages * sizeof(RpptROI));
        roiTensorPtrSrc[0].xywhROI.xy.x = 0;
        roiTensorPtrSrc[0].xywhROI.xy.y = 0;
        roiTensorPtrSrc[0].xywhROI.roiWidth = oSrcSize.width;
        roiTensorPtrSrc[0].xywhROI.roiHeight = oSrcSize.height;
        RpptRoiType roiTypeSrc;
        roiTypeSrc = RpptRoiType::XYWH;
        RpptImagePatch *dstImgSizes;
        hipHostMalloc(&dstImgSizes, noOfImages * sizeof(RpptImagePatch));
        dstImgSizes[0].width = oDstSize.width;
        dstImgSizes[0].height = oDstSize.height;

        srcDescPtr->strides.nStride = srcDescPtr->c * srcDescPtr->w * srcDescPtr->h;
	srcDescPtr->strides.cStride = srcDescPtr->w * srcDescPtr->h;
	srcDescPtr->strides.hStride = srcDescPtr->w;
	srcDescPtr->strides.wStride = 1;

	dstDescPtr->strides.nStride = dstDescPtr->c * dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.cStride = dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.hStride = dstDescPtr->w;
        dstDescPtr->strides.wStride = 1;

        unsigned long long ioBufferSize = 0;
        unsigned long long oBufferSize = 0;

        ioBufferSize = (unsigned long long)srcDescPtr->h * (unsigned long long)srcDescPtr->w * (unsigned long long)srcDescPtr->c * (unsigned long long)noOfImages;
        oBufferSize = (unsigned long long)dstDescPtr->h * (unsigned long long)dstDescPtr->w * (unsigned long long)dstDescPtr->c * (unsigned long long)noOfImages;

        unsigned long long ioBufferSizeInBytes_u8 = ioBufferSize + srcDescPtr->offsetInBytes;
        unsigned long long oBufferSizeInBytes_u8 = oBufferSize + dstDescPtr->offsetInBytes;

        Rpp8u *d_input,*d_output;
        hipMalloc(&d_input,ioBufferSizeInBytes_u8);
        hipMalloc(&d_output,oBufferSizeInBytes_u8);
	Rpp8u *offsetted_input, *offsetted_output;
        offsetted_input = d_input + srcDescPtr->offsetInBytes;
        
	Rpp8u *temp_in;
        hipMalloc(&temp_in, oSrcSize.height * oSrcSize.width * ip_channel);
        for (int i = 0; i < ip_channel; i++)
        {
            hipMemcpy(temp_in + (i * oSrcSize.height * oSrcSize.width), pSrc[i], oSrcSize.height * oSrcSize.width, hipMemcpyDeviceToDevice);
        }

        Rpp32u elementsInRow = oSrcSize.width * srcDescPtr->c;

        for (int j = 0; j < oSrcSize.height; j++)
        {
            hipMemcpy(offsetted_input, temp_in, elementsInRow * sizeof (Rpp8u), hipMemcpyDeviceToDevice);
            temp_in += elementsInRow;
            offsetted_input += srcDescPtr->w * srcDescPtr->c;
        }


        RppStatus status;
        RpptInterpolationType interpolationType = RpptInterpolationType::BILINEAR;

        if(eInterpolation == NPPI_INTER_UNDEFINED) {
                interpolationType = RpptInterpolationType::NEAREST_NEIGHBOR;
        } else if (eInterpolation == NPPI_INTER_LINEAR) {
                interpolationType = RpptInterpolationType::BILINEAR;
        } else if (eInterpolation == NPPI_INTER_CUBIC) {
                interpolationType = RpptInterpolationType::BICUBIC;
        } else if(eInterpolation == NPPI_INTER_LANCZOS) {
                interpolationType = RpptInterpolationType::LANCZOS;
        } else {
                status = RPP_ERROR_INVALID_ARGUMENTS;
                return(hipRppStatusTocudaNppStatus(status));
        }

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);
        clock_t start, end;
        double gpu_time_used;

        start = clock();
        status = rppt_resize_gpu((RppPtr_t)d_input, srcDescPtr, (RppPtr_t)d_output, dstDescPtr, dstImgSizes, interpolationType, roiTensorPtrSrc, roiTypeSrc, handle);
        hipDeviceSynchronize();
        end = clock();

        printf("\nnppiResize_8u_P3R is %d\n", status);
        gpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
        std::cout << "\nGPU Time - Resize_8u : " << gpu_time_used;
        printf("\n");

        rppDestroyGPU(handle);
        Rpp32u elementsInRowMax = dstDescPtr->w * dstDescPtr->c;
        Rpp32u elementsInRowout = dstImgSizes->width * dstDescPtr->c;
        //Rpp32u elementsInRowout = oSrcSize.width * dstDescPtr->c;
        offsetted_output = d_output + dstDescPtr->offsetInBytes;
        Rpp8u *temp_output;
        hipMalloc(&temp_output, oDstSize.height * oDstSize.width * ip_channel);
        for (int k = 0; k < oDstSize.height; k++)
        {
            hipMemcpy(temp_output + (k * elementsInRowout), offsetted_output, elementsInRowout * sizeof (Rpp8u),hipMemcpyDeviceToDevice);
            //temp_output += elementsInRowout;
            offsetted_output += elementsInRowMax;
        }
        for (int m = 0; m < ip_channel; m++)
        {
            hipMemcpy(pDst[m],temp_output + (m * oDstSize.height * oDstSize.width), oDstSize.height * oDstSize.width, hipMemcpyDeviceToDevice);
        }


        hipFree(d_input);
        hipFree(d_output);
        hipHostFree(roiTensorPtrSrc);
        hipHostFree(dstImgSizes);
	hipFree(temp_in);
        hipFree(temp_output);

        return(hipRppStatusTocudaNppStatus(status));
}*/

/*NppStatus nppiResize_8u_C3R(const Npp8u *pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI,
                        Npp8u *pDst, int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI, int eInterpolation)

{
        //pkd
        int noOfImages = 1;
        int ip_channel = 3;
        RpptDesc srcDesc, dstDesc;
        RpptDescPtr srcDescPtr, dstDescPtr;
        srcDescPtr = &srcDesc;
        dstDescPtr = &dstDesc;

        srcDescPtr->layout = RpptLayout::NHWC;
        dstDescPtr->layout = RpptLayout::NHWC;
        srcDescPtr->dataType = RpptDataType::U8;
        dstDescPtr->dataType = RpptDataType::U8;

        srcDescPtr->numDims = 4;
        dstDescPtr->numDims = 4;

        srcDescPtr->offsetInBytes = 64;
        dstDescPtr->offsetInBytes = 0;

        srcDescPtr->n = noOfImages;
        srcDescPtr->h = oSrcSize.height;
        srcDescPtr->w = oSrcSize.width;
        srcDescPtr->c = ip_channel;

        dstDescPtr->n = noOfImages;
        dstDescPtr->h = oSrcSize.height;
        dstDescPtr->w = oSrcSize.width;
        dstDescPtr->c = ip_channel;

    // Optionally set w stride as a multiple of 8 for src/dst

        srcDescPtr->w = ((srcDescPtr->w / 8) * 8) + 8;
        dstDescPtr->w = ((dstDescPtr->w / 8) * 8) + 8;

    // Set n/c/h/w strides for src/dst
        RpptROI *roiTensorPtrSrc;
        hipHostMalloc(&roiTensorPtrSrc, noOfImages * sizeof(RpptROI));
        roiTensorPtrSrc[0].xywhROI.xy.x = 0;
        roiTensorPtrSrc[0].xywhROI.xy.y = 0;
        roiTensorPtrSrc[0].xywhROI.roiWidth = oSrcSize.width;
        roiTensorPtrSrc[0].xywhROI.roiHeight = oSrcSize.height;
        RpptRoiType roiTypeSrc;
        roiTypeSrc = RpptRoiType::XYWH;
        RpptImagePatch *dstImgSizes;
        hipHostMalloc(&dstImgSizes, noOfImages * sizeof(RpptImagePatch));
        dstImgSizes[0].width = oDstSize.width;
        dstImgSizes[0].height = oDstSize.height;
		
	srcDescPtr->strides.nStride = srcDescPtr->c * srcDescPtr->w * srcDescPtr->h;
        srcDescPtr->strides.hStride = srcDescPtr->c * srcDescPtr->w;
        srcDescPtr->strides.wStride = srcDescPtr->c;
        srcDescPtr->strides.cStride = 1;

        dstDescPtr->strides.nStride = dstDescPtr->c * dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.hStride = dstDescPtr->c * dstDescPtr->w;
        dstDescPtr->strides.wStride = dstDescPtr->c;
        dstDescPtr->strides.cStride = 1;

        unsigned long long ioBufferSize = 0;
        unsigned long long oBufferSize = 0;

        ioBufferSize = (unsigned long long)srcDescPtr->h * (unsigned long long)srcDescPtr->w * (unsigned long long)srcDescPtr->c * (unsigned long long)noOfImages;
        oBufferSize = (unsigned long long)dstDescPtr->h * (unsigned long long)dstDescPtr->w * (unsigned long long)dstDescPtr->c * (unsigned long long)noOfImages;

        unsigned long long ioBufferSizeInBytes_u8 = ioBufferSize + srcDescPtr->offsetInBytes;
        unsigned long long oBufferSizeInBytes_u8 = oBufferSize + dstDescPtr->offsetInBytes;
		
	unsigned long long ioBufferlength = oSrcSize.width * oSrcSize.height * ip_channel;
		
	Rpp8u *ip_image = (Rpp8u *)calloc(ioBufferlength, 1);
	hipMemcpy(ip_image, pSrc, ioBufferlength * sizeof(Rpp8u), hipMemcpyDeviceToHost);
		
	Rpp8u *input = (Rpp8u *)calloc(ioBufferSizeInBytes_u8, 1);
	Rpp8u *output = (Rpp8u *)calloc(oBufferSizeInBytes_u8, 1);

        Rpp8u *offsetted_input;
	offsetted_input = input + srcDescPtr->offsetInBytes;
		
	Rpp8u *input_temp;
        input_temp = offsetted_input;
		
	Rpp32u elementsInRow = roiTensorPtrSrc[0].xywhROI.roiWidth * srcDescPtr->c;

        for (int j = 0; j < roiTensorPtrSrc[0].xywhROI.roiHeight; j++)
        {
            memcpy(input_temp, ip_image, elementsInRow * sizeof (Rpp8u));
            ip_image += elementsInRow;
            input_temp += srcDescPtr->strides.hStride;
        }
	
	int *d_input;
	int *d_output;
	hipMalloc(&d_input, ioBufferSizeInBytes_u8);
        hipMalloc(&d_output, oBufferSizeInBytes_u8);
        hipMemcpy(d_input, input, ioBufferSizeInBytes_u8, hipMemcpyHostToDevice);
        hipMemcpy(d_output, output, oBufferSizeInBytes_u8, hipMemcpyHostToDevice);
 
        RppStatus status;
        RpptInterpolationType interpolationType = RpptInterpolationType::BILINEAR;

        if(eInterpolation == NPPI_INTER_UNDEFINED) {
                RpptInterpolationType interpolationType = RpptInterpolationType::NEAREST_NEIGHBOR;
        } else if (eInterpolation == NPPI_INTER_LINEAR) {
                RpptInterpolationType interpolationType = RpptInterpolationType::BILINEAR;
        } else if (eInterpolation == NPPI_INTER_CUBIC) {
                RpptInterpolationType interpolationType = RpptInterpolationType::BICUBIC;
        } else if(eInterpolation == NPPI_INTER_LANCZOS) {
                RpptInterpolationType interpolationType = RpptInterpolationType::LANCZOS;
        } else {
                status = RPP_ERROR_INVALID_ARGUMENTS;
                return(hipRppStatusTocudaNppStatus(status));
        }

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);
        clock_t start, end;
        double gpu_time_used;

        start = clock();
        status = rppt_resize_gpu(d_input, srcDescPtr, d_output, dstDescPtr, dstImgSizes, interpolationType, roiTensorPtrSrc, roiTypeSrc, handle);
        hipDeviceSynchronize();
        end = clock();
		
	printf("\nnppiResize_8u_C3R is %d\n", status);
        gpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
        std::cout << "\nGPU Time - Resize_8u : " << gpu_time_used;
        printf("\n");
		
	hipMemcpy(output, d_output, oBufferSizeInBytes_u8, hipMemcpyDeviceToHost);

        rppDestroyGPU(handle);
        
	Rpp32u elementsInRowMax = dstDescPtr->w * dstDescPtr->c;

	Rpp8u *offsetted_output;
	offsetted_output = output + dstDescPtr->offsetInBytes;	
		
	int height = dstImgSizes[0].height;
        int width = dstImgSizes[0].width;

        int op_size = height * width * dstDescPtr->c;
        Rpp8u *temp_output = (Rpp8u *)calloc(op_size, sizeof(Rpp8u));
        Rpp8u *temp_output_row;
        temp_output_row = temp_output;
        Rpp32u elementsInRowout = width * dstDescPtr->c;
        Rpp8u *output_row = offsetted_output;

        for (int k = 0; k < height; k++)
        {
            memcpy(temp_output_row, (output_row), elementsInRowout * sizeof (Rpp8u));
            temp_output_row += elementsInRowout;
            output_row += elementsInRowMax;
        }

	//cv::Mat mat_op_image;
        //mat_op_image = cv::Mat(height, width, CV_8UC3, temp_output);
        //imwrite("./resize01.jpg", mat_op_image);
		
	hipMemcpy(pDst, temp_output, op_size, hipMemcpyHostToDevice);

	hipHostFree(roiTensorPtrSrc);
        hipHostFree(dstImgSizes);
        free(input);
        free(output);
	hipFree(d_input);
        hipFree(d_output);

        return(hipRppStatusTocudaNppStatus(status));
}*/

/******************** resize_crop_mirror ********************/

RppStatus rppt_resize_crop_mirror_gpu(RppPtr_t srcPtr,
                                      RpptDescPtr srcDescPtr,
                                      RppPtr_t dstPtr,
                                      RpptDescPtr dstDescPtr,
                                      RpptImagePatchPtr dstImgSizes,
                                      RpptInterpolationType interpolationType,
                                      Rpp32u *mirrorTensor,
                                      RpptROIPtr roiTensorPtrSrc,
                                      RpptRoiType roiType,
                                      rppHandle_t rppHandle)
{
#ifdef HIP_COMPILE
    if (interpolationType != RpptInterpolationType::BILINEAR)
        return RPP_ERROR_NOT_IMPLEMENTED;

    copy_param_uint(mirrorTensor, rpp::deref(rppHandle), 0);

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        hip_exec_resize_crop_mirror_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                           srcDescPtr,
                                           static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                           dstDescPtr,
                                           dstImgSizes,
                                           interpolationType,
                                           roiTensorPtrSrc,
                                           roiType,
                                           rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        hip_exec_resize_crop_mirror_tensor((half*)(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                           srcDescPtr,
                                           (half*)(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                           dstDescPtr,
                                           dstImgSizes,
                                           interpolationType,
                                           roiTensorPtrSrc,
                                           roiType,
                                           rpp::deref(rppHandle));
    }

    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        hip_exec_resize_crop_mirror_tensor((Rpp32f*)(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                           srcDescPtr,
                                           (Rpp32f*)(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                           dstDescPtr,
                                           dstImgSizes,
                                           interpolationType,
                                           roiTensorPtrSrc,
                                           roiType,
                                           rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        hip_exec_resize_crop_mirror_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                           srcDescPtr,
                                           static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                           dstDescPtr,
                                           dstImgSizes,
                                           interpolationType,
                                           roiTensorPtrSrc,
                                           roiType,
                                           rpp::deref(rppHandle));
}

return RPP_SUCCESS;
#elif defined(OCL_COMPILE)
    return RPP_ERROR_NOT_IMPLEMENTED;
#endif // backend
}

/******************** rotate ********************/

RppStatus rppt_rotate_gpu(RppPtr_t srcPtr,
                          RpptDescPtr srcDescPtr,
                          RppPtr_t dstPtr,
                          RpptDescPtr dstDescPtr,
                          Rpp32f *angle,
                          RpptInterpolationType interpolationType,
                          RpptROIPtr roiTensorPtrSrc,
                          RpptRoiType roiType,
                          rppHandle_t rppHandle)
{
#ifdef HIP_COMPILE
    if ((interpolationType != RpptInterpolationType::BILINEAR) && (interpolationType != RpptInterpolationType::NEAREST_NEIGHBOR))
        return RPP_ERROR_NOT_IMPLEMENTED;

    // Compute affine transformation matrix from rotate angle
    Rpp32f *affineTensor = rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.scratchBufferHost;
    for(int idx = 0; idx < srcDescPtr->n; idx++)
    {
        Rpp32f angleInRad = angle[idx] * PI_OVER_180;
        Rpp32f alpha, beta;
        sincosf(angleInRad, &beta, &alpha);
        ((Rpp32f6 *)affineTensor)[idx] = {alpha, -beta, 0, beta, alpha, 0};
    }

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        hip_exec_warp_affine_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                    srcDescPtr,
                                    static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                    dstDescPtr,
                                    affineTensor,
                                    interpolationType,
                                    roiTensorPtrSrc,
                                    roiType,
                                    rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        hip_exec_warp_affine_tensor((half*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                    srcDescPtr,
                                    (half*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                    dstDescPtr,
                                    affineTensor,
                                    interpolationType,
                                    roiTensorPtrSrc,
                                    roiType,
                                    rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        hip_exec_warp_affine_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                    srcDescPtr,
                                    (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                    dstDescPtr,
                                    affineTensor,
                                    interpolationType,
                                    roiTensorPtrSrc,
                                    roiType,
                                    rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        hip_exec_warp_affine_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                    srcDescPtr,
                                    static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                    dstDescPtr,
                                    affineTensor,
                                    interpolationType,
                                    roiTensorPtrSrc,
                                    roiType,
                                    rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
#elif defined(OCL_COMPILE)
    return RPP_ERROR_NOT_IMPLEMENTED;
#endif // backend
}

/******************** phase ********************/

RppStatus rppt_phase_gpu(RppPtr_t srcPtr1,
                         RppPtr_t srcPtr2,
                         RpptDescPtr srcDescPtr,
                         RppPtr_t dstPtr,
                         RpptDescPtr dstDescPtr,
                         RpptROIPtr roiTensorPtrSrc,
                         RpptRoiType roiType,
                         rppHandle_t rppHandle)
{
#ifdef HIP_COMPILE

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        hip_exec_phase_tensor(static_cast<Rpp8u*>(srcPtr1) + srcDescPtr->offsetInBytes,
                              static_cast<Rpp8u*>(srcPtr2) + srcDescPtr->offsetInBytes,
                              srcDescPtr,
                              static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                              dstDescPtr,
                              roiTensorPtrSrc,
                              roiType,
                              rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        hip_exec_phase_tensor(reinterpret_cast<half*>(static_cast<Rpp8u*>(srcPtr1) + srcDescPtr->offsetInBytes),
                              reinterpret_cast<half*>(static_cast<Rpp8u*>(srcPtr2) + srcDescPtr->offsetInBytes),
                              srcDescPtr,
                              reinterpret_cast<half*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                              dstDescPtr,
                              roiTensorPtrSrc,
                              roiType,
                              rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        hip_exec_phase_tensor(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr1) + srcDescPtr->offsetInBytes),
                              reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr2) + srcDescPtr->offsetInBytes),
                              srcDescPtr,
                              reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                              dstDescPtr,
                              roiTensorPtrSrc,
                              roiType,
                              rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        hip_exec_phase_tensor(static_cast<Rpp8s*>(srcPtr1) + srcDescPtr->offsetInBytes,
                              static_cast<Rpp8s*>(srcPtr2) + srcDescPtr->offsetInBytes,
                              srcDescPtr,
                              static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                              dstDescPtr,
                              roiTensorPtrSrc,
                              roiType,
                              rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
#elif defined(OCL_COMPILE)
    return RPP_ERROR_NOT_IMPLEMENTED;
#endif // backend
}

/******************** slice ********************/

RppStatus rppt_slice_gpu(RppPtr_t srcPtr,
                         RpptGenericDescPtr srcGenericDescPtr,
                         RppPtr_t dstPtr,
                         RpptGenericDescPtr dstGenericDescPtr,
                         RpptROI3DPtr roiGenericPtrSrc,
                         RpptRoi3DType roiType,
                         rppHandle_t rppHandle)
{
#ifdef HIP_COMPILE
    if ((srcGenericDescPtr->layout != RpptLayout::NCDHW) && (srcGenericDescPtr->layout != RpptLayout::NDHWC)) return RPP_ERROR_INVALID_SRC_LAYOUT;
    if ((dstGenericDescPtr->layout != RpptLayout::NCDHW) && (dstGenericDescPtr->layout != RpptLayout::NDHWC)) return RPP_ERROR_INVALID_DST_LAYOUT;
    if (srcGenericDescPtr->layout != dstGenericDescPtr->layout) return RPP_ERROR_INVALID_ARGUMENTS;
    if ((srcGenericDescPtr->dataType != RpptDataType::F32) && (srcGenericDescPtr->dataType != RpptDataType::U8)) return RPP_ERROR_INVALID_SRC_DATATYPE;
    if ((dstGenericDescPtr->dataType != RpptDataType::F32) && (dstGenericDescPtr->dataType != RpptDataType::U8)) return RPP_ERROR_INVALID_DST_DATATYPE;

    if ((srcGenericDescPtr->dataType == RpptDataType::F32) && (dstGenericDescPtr->dataType == RpptDataType::F32))
    {
        hip_exec_slice_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcGenericDescPtr->offsetInBytes),
                              srcGenericDescPtr,
                              (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstGenericDescPtr->offsetInBytes),
                              dstGenericDescPtr,
                              roiGenericPtrSrc,
                              rpp::deref(rppHandle));
    }
    else if ((srcGenericDescPtr->dataType == RpptDataType::U8) && (dstGenericDescPtr->dataType == RpptDataType::U8))
    {
        hip_exec_slice_tensor(static_cast<Rpp8u*>(srcPtr) + srcGenericDescPtr->offsetInBytes,
                              srcGenericDescPtr,
                              static_cast<Rpp8u*>(dstPtr) + dstGenericDescPtr->offsetInBytes,
                              dstGenericDescPtr,
                              roiGenericPtrSrc,
                              rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
#elif defined(OCL_COMPILE)
    return RPP_ERROR_NOT_IMPLEMENTED;
#endif // backend
}

RppStatus rppt_crop_and_patch_gpu(RppPtr_t srcPtr1,
                                  RppPtr_t srcPtr2,
                                  RpptDescPtr srcDescPtr,
                                  RppPtr_t dstPtr,
                                  RpptDescPtr dstDescPtr,
                                  RpptROIPtr roiTensorPtrSrc,
                                  RpptROIPtr cropTensorPtr,
                                  RpptROIPtr patchTensorPtr,
                                  RpptRoiType roiType,
                                  rppHandle_t rppHandle)
{
#ifdef HIP_COMPILE
    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        hip_exec_crop_and_patch_tensor(static_cast<Rpp8u*>(srcPtr1) + srcDescPtr->offsetInBytes,
                                       static_cast<Rpp8u*>(srcPtr2) + srcDescPtr->offsetInBytes,
                                       srcDescPtr,
                                       static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                       dstDescPtr,
                                       roiTensorPtrSrc,
                                       cropTensorPtr,
                                       patchTensorPtr,
                                       roiType,
                                       rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        hip_exec_crop_and_patch_tensor(reinterpret_cast<half*>(static_cast<Rpp8u*>(srcPtr1)) + srcDescPtr->offsetInBytes,
                                       reinterpret_cast<half*>(static_cast<Rpp8u*>(srcPtr2)) + srcDescPtr->offsetInBytes,
                                       srcDescPtr,
                                       reinterpret_cast<half*>(static_cast<Rpp8u*>(dstPtr)) + dstDescPtr->offsetInBytes,
                                       dstDescPtr,
                                       roiTensorPtrSrc,
                                       cropTensorPtr,
                                       patchTensorPtr,
                                       roiType,
                                       rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        hip_exec_crop_and_patch_tensor(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr1)) + srcDescPtr->offsetInBytes,
                                       reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr2)) + srcDescPtr->offsetInBytes,
                                       srcDescPtr,
                                       reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr)) + dstDescPtr->offsetInBytes,
                                       dstDescPtr,
                                       roiTensorPtrSrc,
                                       cropTensorPtr,
                                       patchTensorPtr,
                                       roiType,
                                       rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        hip_exec_crop_and_patch_tensor(static_cast<Rpp8s*>(srcPtr1) + srcDescPtr->offsetInBytes,
                                       static_cast<Rpp8s*>(srcPtr2) + srcDescPtr->offsetInBytes,
                                       srcDescPtr,
                                       static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                       dstDescPtr,
                                       roiTensorPtrSrc,
                                       cropTensorPtr,
                                       patchTensorPtr,
                                       roiType,
                                       rpp::deref(rppHandle));
    }
    return RPP_SUCCESS;
#elif defined(OCL_COMPILE)
    return RPP_ERROR_NOT_IMPLEMENTED;
#endif // backend
}

/******************** flip_voxel ********************/

RppStatus rppt_flip_voxel_gpu(RppPtr_t srcPtr,
                              RpptGenericDescPtr srcGenericDescPtr,
                              RppPtr_t dstPtr,
                              RpptGenericDescPtr dstGenericDescPtr,
                              Rpp32u *horizontalTensor,
                              Rpp32u *verticalTensor,
                              Rpp32u *depthTensor,
                              RpptROI3DPtr roiGenericPtrSrc,
                              RpptRoi3DType roiType,
                              rppHandle_t rppHandle)
{
#ifdef HIP_COMPILE
    if ((srcGenericDescPtr->layout != RpptLayout::NCDHW) && (srcGenericDescPtr->layout != RpptLayout::NDHWC)) return RPP_ERROR_INVALID_SRC_LAYOUT;
    if ((dstGenericDescPtr->layout != RpptLayout::NCDHW) && (dstGenericDescPtr->layout != RpptLayout::NDHWC)) return RPP_ERROR_INVALID_DST_LAYOUT;
    if (srcGenericDescPtr->layout != dstGenericDescPtr->layout) return RPP_ERROR_INVALID_ARGUMENTS;
    if ((srcGenericDescPtr->dataType != RpptDataType::F32) && (srcGenericDescPtr->dataType != RpptDataType::U8)) return RPP_ERROR_INVALID_SRC_DATATYPE;
    if ((dstGenericDescPtr->dataType != RpptDataType::F32) && (dstGenericDescPtr->dataType != RpptDataType::U8)) return RPP_ERROR_INVALID_DST_DATATYPE;

    if ((srcGenericDescPtr->dataType == RpptDataType::F32) && (dstGenericDescPtr->dataType == RpptDataType::F32))
    {
        hip_exec_flip_voxel_tensor(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcGenericDescPtr->offsetInBytes),
                                   srcGenericDescPtr,
                                   reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstGenericDescPtr->offsetInBytes),
                                   dstGenericDescPtr,
                                   roiGenericPtrSrc,
                                   horizontalTensor,
                                   verticalTensor,
                                   depthTensor,
                                   roiType,
                                   rpp::deref(rppHandle));
    }
    else if ((srcGenericDescPtr->dataType == RpptDataType::U8) && (dstGenericDescPtr->dataType == RpptDataType::U8))
    {
        hip_exec_flip_voxel_tensor(static_cast<Rpp8u*>(srcPtr) + srcGenericDescPtr->offsetInBytes,
                                   srcGenericDescPtr,
                                   static_cast<Rpp8u*>(dstPtr) + dstGenericDescPtr->offsetInBytes,
                                   dstGenericDescPtr,
                                   roiGenericPtrSrc,
                                   horizontalTensor,
                                   verticalTensor,
                                   depthTensor,
                                   roiType,
                                   rpp::deref(rppHandle));
    }
    return RPP_SUCCESS;
#elif defined(OCL_COMPILE)
    return RPP_ERROR_NOT_IMPLEMENTED;
#endif // backend
}

/******************** remap ********************/
RppStatus rppt_remap_gpu(RppPtr_t srcPtr,
                         RpptDescPtr srcDescPtr,
                         RppPtr_t dstPtr,
                         RpptDescPtr dstDescPtr,
                         Rpp32f *rowRemapTable,
                         Rpp32f *colRemapTable,
                         RpptDescPtr tableDescPtr,
                         RpptInterpolationType interpolationType,
                         RpptROIPtr roiTensorPtrSrc,
                         RpptRoiType roiType,
                         rppHandle_t rppHandle)
{
#ifdef HIP_COMPILE
    if (interpolationType != RpptInterpolationType::NEAREST_NEIGHBOR && interpolationType != RpptInterpolationType::BILINEAR)
        return RPP_ERROR_NOT_IMPLEMENTED;

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        hip_exec_remap_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                              srcDescPtr,
                              static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                              dstDescPtr,
                              rowRemapTable,
                              colRemapTable,
                              tableDescPtr,
                              interpolationType,
                              roiTensorPtrSrc,
                              roiType,
                              rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        hip_exec_remap_tensor(reinterpret_cast<half*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                              srcDescPtr,
                              reinterpret_cast<half*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                              dstDescPtr,
                              rowRemapTable,
                              colRemapTable,
                              tableDescPtr,
                              interpolationType,
                              roiTensorPtrSrc,
                              roiType,
                              rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        hip_exec_remap_tensor(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                              srcDescPtr,
                              reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                              dstDescPtr,
                              rowRemapTable,
                              colRemapTable,
                              tableDescPtr,
                              interpolationType,
                              roiTensorPtrSrc,
                              roiType,
                              rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        hip_exec_remap_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                              srcDescPtr,
                              static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                              dstDescPtr,
                              rowRemapTable,
                              colRemapTable,
                              tableDescPtr,
                              interpolationType,
                              roiTensorPtrSrc,
                              roiType,
                              rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::U16) && (dstDescPtr->dataType == RpptDataType::U16))
    {
        hip_exec_remap_tensor(reinterpret_cast<Rpp16u*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                              srcDescPtr,
                              reinterpret_cast<Rpp16u*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                              dstDescPtr,
                              rowRemapTable,
                              colRemapTable,
                              tableDescPtr,
                              interpolationType,
                              roiTensorPtrSrc,
                              roiType,
                              rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::S16) && (dstDescPtr->dataType == RpptDataType::S16))
    {
        hip_exec_remap_tensor(reinterpret_cast<Rpp16s*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                              srcDescPtr,
                              reinterpret_cast<Rpp16s*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                              dstDescPtr,
                              rowRemapTable,
                              colRemapTable,
                              tableDescPtr,
                              interpolationType,
                              roiTensorPtrSrc,
                              roiType,
                              rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F64) && (dstDescPtr->dataType == RpptDataType::F64))
    {
        hip_exec_remap_tensor(reinterpret_cast<Rpp64f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                              srcDescPtr,
                              reinterpret_cast<Rpp64f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                              dstDescPtr,
                              rowRemapTable,
                              colRemapTable,
                              tableDescPtr,
                              interpolationType,
                              roiTensorPtrSrc,
                              roiType,
                              rpp::deref(rppHandle));
    }
    return RPP_SUCCESS;
#elif defined(OCL_COMPILE)
    return RPP_ERROR_NOT_IMPLEMENTED;
#endif // backend
}


NppStatus nppiRemap_8u_C1R_Ctx(const Npp8u  * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                     const Npp32f * pXMap, int nXMapStep, const Npp32f * pYMap, int nYMapStep,
                           Npp8u  * pDst, int nDstStep, NppiSize oDstSizeROI, int eInterpolation, NppStreamContext nppStreamCtx)
{
    //pkd
    int noOfImages = 1;
    int ip_channel = 1;
    RpptDesc srcDesc, dstDesc;
    RpptDescPtr srcDescPtr = &srcDesc;
    RpptDescPtr dstDescPtr = &dstDesc;

    
    srcDescPtr->layout      = RpptLayout::NCHW;
    srcDescPtr->dataType    = RpptDataType::U8;
    srcDescPtr->numDims     = 4;
    srcDescPtr->offsetInBytes = 0;
    srcDescPtr->n           = noOfImages;
    srcDescPtr->c           = ip_channel;
    srcDescPtr->h           = oSrcSize.height;
    srcDescPtr->w           = oSrcSize.width;

    dstDescPtr->layout      = RpptLayout::NCHW;
    dstDescPtr->dataType    = RpptDataType::U8;
    dstDescPtr->numDims     = 4;
    dstDescPtr->offsetInBytes = 0;
    dstDescPtr->n           = noOfImages;
    dstDescPtr->c           = ip_channel;
    dstDescPtr->h           = oDstSizeROI.height;
    dstDescPtr->w           = oDstSizeROI.width;


    // ROI for source
    RpptROI *roiTensorPtrSrc;
    hipHostMalloc(&roiTensorPtrSrc, noOfImages * sizeof(RpptROI));
    roiTensorPtrSrc[0].xywhROI.xy.x     = oSrcROI.x;
    roiTensorPtrSrc[0].xywhROI.xy.y     = oSrcROI.y;
    roiTensorPtrSrc[0].xywhROI.roiWidth  = oSrcROI.width;
    roiTensorPtrSrc[0].xywhROI.roiHeight = oSrcROI.height;
    RpptRoiType roiTypeSrc = RpptRoiType::XYWH;

    // Destination image patch
    RpptImagePatch *dstImgSizes;
    hipHostMalloc(&dstImgSizes, noOfImages * sizeof(RpptImagePatch));
    dstImgSizes[0].width  = oDstSizeROI.width;
    dstImgSizes[0].height = oDstSizeROI.height;

    // Strides for NCHW: nStride = c*h*w, cStride = h*w, hStride = w, wStride = 1
    srcDescPtr->strides.nStride = srcDescPtr->c * srcDescPtr->h * srcDescPtr->w;
    srcDescPtr->strides.cStride = srcDescPtr->h * srcDescPtr->w;
    srcDescPtr->strides.hStride = srcDescPtr->w;
    srcDescPtr->strides.wStride = 1;

    dstDescPtr->strides.nStride = dstDescPtr->c * dstDescPtr->h * dstDescPtr->w;
    dstDescPtr->strides.cStride = dstDescPtr->h * dstDescPtr->w;
    dstDescPtr->strides.hStride = dstDescPtr->w;
    dstDescPtr->strides.wStride = 1;

    RpptDesc tableDesc;
    RpptDescPtr tableDescPtr = &tableDesc;
	
    tableDescPtr->layout      = RpptLayout::NCHW;
    tableDescPtr->dataType    = RpptDataType::U8;
    tableDescPtr->numDims     = 4;
    tableDescPtr->offsetInBytes = 0;
    tableDescPtr->n           = noOfImages;
    tableDescPtr->c           = 1;
    tableDescPtr->h           = oDstSizeROI.height;
    tableDescPtr->w           = oDstSizeROI.width;
    tableDescPtr->strides.nStride = tableDescPtr->c * tableDescPtr->h * tableDescPtr->w;
    tableDescPtr->strides.cStride = tableDescPtr->h * tableDescPtr->w;
    tableDescPtr->strides.hStride = tableDescPtr->w;
    tableDescPtr->strides.wStride = 1;
    
    unsigned long long inBufSize  = (unsigned long long)srcDescPtr->n * srcDescPtr->c * srcDescPtr->h * srcDescPtr->w * sizeof(Rpp8u);
    unsigned long long outBufSize = (unsigned long long)dstDescPtr->n * dstDescPtr->c * dstDescPtr->h * dstDescPtr->w * sizeof(Rpp8u);
    Rpp8u *d_input,*d_output;
	hipMalloc(&d_input,  inBufSize);
    hipMalloc(&d_output, outBufSize);

    // Select interpolation type
	RppStatus status;
    RpptInterpolationType interpolationType = RpptInterpolationType::BILINEAR;
    if(eInterpolation == NPPI_INTER_NN) {
           interpolationType = RpptInterpolationType::NEAREST_NEIGHBOR;
    } else if (eInterpolation == NPPI_INTER_LINEAR) {
           interpolationType = RpptInterpolationType::BILINEAR;
    } else if (eInterpolation == NPPI_INTER_CUBIC) {
           interpolationType = RpptInterpolationType::BICUBIC;
    } else if(eInterpolation == NPPI_INTER_LANCZOS) {
           interpolationType = RpptInterpolationType::LANCZOS;
    } else {
           status = RPP_ERROR_INVALID_ARGUMENTS;
		   return(hipRppStatusTocudaNppStatus(status));
    }


    Rpp8u *d_off_in = d_input;
    for(int j = 0; j < srcDescPtr->h; j++) {
        const Rpp8u* srcRow = (const Rpp8u*)((const char*)pSrc + j * nSrcStep);
        hipMemcpy(d_off_in, srcRow, srcDescPtr->w * sizeof(Rpp8u), hipMemcpyDeviceToDevice);
        d_off_in += srcDescPtr->w;
    }
    rppHandle_t handle;
    hipStream_t stream;
    hipStreamCreate(&stream);
    rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);
    status = rppt_remap_gpu(d_input, srcDescPtr, d_output, dstDescPtr, (Rpp32f *)pYMap, (Rpp32f *)pXMap, tableDescPtr, 
							interpolationType, roiTensorPtrSrc, roiTypeSrc, handle);
    hipDeviceSynchronize();
    rppDestroyGPU(handle);
    // Copy output back row by row
    Rpp8u *d_off_out = d_output;
    for(int k = 0; k < dstDescPtr->h; k++) {
        Rpp8u* dstRow = (Rpp8u*)((char*)pDst + k * nDstStep);
        hipMemcpy(dstRow, d_off_out, dstDescPtr->w * sizeof(Rpp8u), hipMemcpyDeviceToDevice);
        d_off_out += dstDescPtr->w;
    }

    hipFree(d_input);
    hipFree(d_output);
    hipHostFree(roiTensorPtrSrc);
    hipHostFree(dstImgSizes);
    return(hipRppStatusTocudaNppStatus(status));
}


NppStatus nppiRemap_8u_C3R_Ctx(const Npp8u  * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                     const Npp32f * pXMap, int nXMapStep, const Npp32f * pYMap, int nYMapStep,
                           Npp8u  * pDst, int nDstStep, NppiSize oDstSizeROI, int eInterpolation, NppStreamContext nppStreamCtx)
{
    // Three-channel implementation
    int noOfImages  = 1;
    int ip_channel  = 3;
    RpptDesc srcDesc, dstDesc;
    RpptDescPtr srcDescPtr = &srcDesc;
    RpptDescPtr dstDescPtr = &dstDesc;

    // Use NHWC layout for interleaved C3R
    srcDescPtr->layout      = RpptLayout::NHWC;
    srcDescPtr->dataType    = RpptDataType::U8;
    srcDescPtr->numDims     = 4;
    srcDescPtr->offsetInBytes = 0;
    srcDescPtr->n           = noOfImages;
    srcDescPtr->h           = oSrcSize.height;
    srcDescPtr->w           = oSrcSize.width;
    srcDescPtr->c           = ip_channel;

    dstDescPtr->layout      = RpptLayout::NHWC;
    dstDescPtr->dataType    = RpptDataType::U8;
    dstDescPtr->numDims     = 4;
    dstDescPtr->offsetInBytes = 0;
    dstDescPtr->n           = noOfImages;
    dstDescPtr->h           = oDstSizeROI.height;
    dstDescPtr->w           = oDstSizeROI.width;
    dstDescPtr->c           = ip_channel;

    // Source ROI
    RpptROI *roiTensorPtrSrc;
    hipHostMalloc(&roiTensorPtrSrc, noOfImages * sizeof(RpptROI));
    roiTensorPtrSrc[0].xywhROI.xy.x     = oSrcROI.x;
    roiTensorPtrSrc[0].xywhROI.xy.y     = oSrcROI.y;
    roiTensorPtrSrc[0].xywhROI.roiWidth  = oSrcROI.width;
    roiTensorPtrSrc[0].xywhROI.roiHeight = oSrcROI.height;
    RpptRoiType roiTypeSrc = RpptRoiType::XYWH;

    // Destination patch
    RpptImagePatch *dstImgSizes;
    hipHostMalloc(&dstImgSizes, noOfImages * sizeof(RpptImagePatch));
    dstImgSizes[0].width  = oDstSizeROI.width;
    dstImgSizes[0].height = oDstSizeROI.height;

    // Strides for NHWC: nStride = h*w*c, hStride = w*c, wStride = c, cStride = 1
    srcDescPtr->strides.nStride = srcDescPtr->h * srcDescPtr->w * srcDescPtr->c;
    srcDescPtr->strides.hStride = srcDescPtr->w * srcDescPtr->c;
    srcDescPtr->strides.wStride = srcDescPtr->c;
    srcDescPtr->strides.cStride = 1;

    dstDescPtr->strides.nStride = dstDescPtr->h * dstDescPtr->w * dstDescPtr->c;
    dstDescPtr->strides.hStride = dstDescPtr->w * dstDescPtr->c;
    dstDescPtr->strides.wStride = dstDescPtr->c;
    dstDescPtr->strides.cStride = 1;

    // Table descriptor remains single-channel
    RpptDesc tableDesc;
    RpptDescPtr tableDescPtr = &tableDesc;
    tableDescPtr->layout      = RpptLayout::NHWC;
    tableDescPtr->dataType    = RpptDataType::U8;
    tableDescPtr->numDims     = 4;
    tableDescPtr->offsetInBytes = 0;
    tableDescPtr->n           = noOfImages;
    tableDescPtr->h           = oDstSizeROI.height;
    tableDescPtr->w           = oDstSizeROI.width;
    tableDescPtr->c           = 1;
    tableDescPtr->strides.nStride = tableDescPtr->h * tableDescPtr->w * tableDescPtr->c;
    tableDescPtr->strides.hStride = tableDescPtr->w * tableDescPtr->c;
    tableDescPtr->strides.wStride = tableDescPtr->c;
    tableDescPtr->strides.cStride = 1;

    // Allocate device buffers
    size_t inBufSize  = srcDescPtr->n * srcDescPtr->h * srcDescPtr->w * srcDescPtr->c * sizeof(Rpp8u);
    size_t outBufSize = dstDescPtr->n * dstDescPtr->h * dstDescPtr->w * dstDescPtr->c * sizeof(Rpp8u);
    Rpp8u *d_input, *d_output;
    hipMalloc(&d_input,  inBufSize);
    hipMalloc(&d_output, outBufSize);

    // Choose interpolation
    RpptInterpolationType interpolationType;
    switch(eInterpolation) {
        case NPPI_INTER_NN:      interpolationType = RpptInterpolationType::NEAREST_NEIGHBOR; break;
        case NPPI_INTER_LINEAR:  interpolationType = RpptInterpolationType::BILINEAR;        break;
        case NPPI_INTER_CUBIC:   interpolationType = RpptInterpolationType::BICUBIC;         break;
        case NPPI_INTER_LANCZOS: interpolationType = RpptInterpolationType::LANCZOS;         break;
        default:
            return hipRppStatusTocudaNppStatus(RPP_ERROR_INVALID_ARGUMENTS);
    }

    // Copy host->device row by row
    Rpp8u *d_off_in = d_input;
    for(int y = 0; y < srcDescPtr->h; y++) {
        const Rpp8u* srcRow = (const Rpp8u*)((const char*)pSrc + y * nSrcStep);
        hipMemcpy(d_off_in, srcRow, srcDescPtr->w * srcDescPtr->c * sizeof(Rpp8u), hipMemcpyDeviceToDevice);
        d_off_in += srcDescPtr->w * srcDescPtr->c;
    }

    // Run remap
    rppHandle_t handle;
    hipStream_t stream;
    hipStreamCreate(&stream);
    rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);
    RppStatus status = rppt_remap_gpu(
        d_input, srcDescPtr,
        d_output, dstDescPtr,
        (Rpp32f*)pYMap, (Rpp32f*)pXMap, tableDescPtr,
        interpolationType, roiTensorPtrSrc, roiTypeSrc,
        handle
    );
    hipDeviceSynchronize();
    rppDestroyGPU(handle);

    // Copy device->host
    Rpp8u *d_off_out = d_output;
    for(int y = 0; y < dstDescPtr->h; y++) {
        Rpp8u* dstRow = (Rpp8u*)((char*)pDst + y * nDstStep);
        hipMemcpy(dstRow, d_off_out, dstDescPtr->w * dstDescPtr->c * sizeof(Rpp8u), hipMemcpyDeviceToDevice);
        d_off_out += dstDescPtr->w * dstDescPtr->c;
    }

    hipFree(d_input);
    hipFree(d_output);
    hipHostFree(roiTensorPtrSrc);
    hipHostFree(dstImgSizes);

    return hipRppStatusTocudaNppStatus(status);
}

NppStatus nppiRemap_16u_C1R_Ctx(const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                const Npp32f *pXMap, int nXMapStep, const Npp32f *pYMap, int nYMapStep,
                                      Npp16u *pDst, int nDstStep, NppiSize oDstSizeROI, int eInterpolation, NppStreamContext nppStreamCtx)
{
    //pkd
    int noOfImages = 1;
    int ip_channel = 1;
    RpptDesc srcDesc, dstDesc;
    RpptDescPtr srcDescPtr = &srcDesc;
    RpptDescPtr dstDescPtr = &dstDesc;

    
    srcDescPtr->layout      = RpptLayout::NCHW;
    srcDescPtr->dataType    = RpptDataType::U16;
    srcDescPtr->numDims     = 4;
    srcDescPtr->offsetInBytes = 0;
    srcDescPtr->n           = noOfImages;
    srcDescPtr->c           = ip_channel;
    srcDescPtr->h           = oSrcSize.height;
    srcDescPtr->w           = oSrcSize.width;

    dstDescPtr->layout      = RpptLayout::NCHW;
    dstDescPtr->dataType    = RpptDataType::U16;
    dstDescPtr->numDims     = 4;
    dstDescPtr->offsetInBytes = 0;
    dstDescPtr->n           = noOfImages;
    dstDescPtr->c           = ip_channel;
    dstDescPtr->h           = oDstSizeROI.height;
    dstDescPtr->w           = oDstSizeROI.width;


    // ROI for source
    RpptROI *roiTensorPtrSrc;
    hipHostMalloc(&roiTensorPtrSrc, noOfImages * sizeof(RpptROI));
    roiTensorPtrSrc[0].xywhROI.xy.x     = oSrcROI.x;
    roiTensorPtrSrc[0].xywhROI.xy.y     = oSrcROI.y;
    roiTensorPtrSrc[0].xywhROI.roiWidth  = oSrcROI.width;
    roiTensorPtrSrc[0].xywhROI.roiHeight = oSrcROI.height;
    RpptRoiType roiTypeSrc = RpptRoiType::XYWH;

    // Destination image patch
    RpptImagePatch *dstImgSizes;
    hipHostMalloc(&dstImgSizes, noOfImages * sizeof(RpptImagePatch));
    dstImgSizes[0].width  = oDstSizeROI.width;
    dstImgSizes[0].height = oDstSizeROI.height;

    // Strides for NCHW: nStride = c*h*w, cStride = h*w, hStride = w, wStride = 1
    srcDescPtr->strides.nStride = srcDescPtr->c * srcDescPtr->h * srcDescPtr->w;
    srcDescPtr->strides.cStride = srcDescPtr->h * srcDescPtr->w;
    srcDescPtr->strides.hStride = srcDescPtr->w;
    srcDescPtr->strides.wStride = 1;

    dstDescPtr->strides.nStride = dstDescPtr->c * dstDescPtr->h * dstDescPtr->w;
    dstDescPtr->strides.cStride = dstDescPtr->h * dstDescPtr->w;
    dstDescPtr->strides.hStride = dstDescPtr->w;
    dstDescPtr->strides.wStride = 1;

    RpptDesc tableDesc;
    RpptDescPtr tableDescPtr = &tableDesc;
	
    tableDescPtr->layout      = RpptLayout::NCHW;
    tableDescPtr->dataType    = RpptDataType::U16;
    tableDescPtr->numDims     = 4;
    tableDescPtr->offsetInBytes = 0;
    tableDescPtr->n           = noOfImages;
    tableDescPtr->c           = 1;
    tableDescPtr->h           = oDstSizeROI.height;
    tableDescPtr->w           = oDstSizeROI.width;
    tableDescPtr->strides.nStride = tableDescPtr->c * tableDescPtr->h * tableDescPtr->w;
    tableDescPtr->strides.cStride = tableDescPtr->h * tableDescPtr->w;
    tableDescPtr->strides.hStride = tableDescPtr->w;
    tableDescPtr->strides.wStride = 1;
    
    unsigned long long inBufSize  = (unsigned long long)srcDescPtr->n * srcDescPtr->c * srcDescPtr->h * srcDescPtr->w * sizeof(Rpp16u);
    unsigned long long outBufSize = (unsigned long long)dstDescPtr->n * dstDescPtr->c * dstDescPtr->h * dstDescPtr->w * sizeof(Rpp16u);
    Rpp16u *d_input,*d_output;
	hipMalloc(&d_input,  inBufSize);
    hipMalloc(&d_output, outBufSize);

    // Select interpolation type
	RppStatus status;
    RpptInterpolationType interpolationType = RpptInterpolationType::BILINEAR;
    if(eInterpolation == NPPI_INTER_NN) {
           interpolationType = RpptInterpolationType::NEAREST_NEIGHBOR;
    } else if (eInterpolation == NPPI_INTER_LINEAR) {
           interpolationType = RpptInterpolationType::BILINEAR;
    } else if (eInterpolation == NPPI_INTER_CUBIC) {
           interpolationType = RpptInterpolationType::BICUBIC;
    } else if(eInterpolation == NPPI_INTER_LANCZOS) {
           interpolationType = RpptInterpolationType::LANCZOS;
    } else {
           status = RPP_ERROR_INVALID_ARGUMENTS;
		   return(hipRppStatusTocudaNppStatus(status));
    }


    Rpp16u *d_off_in = d_input;
    for(int j = 0; j < srcDescPtr->h; j++) {
        const Rpp16u* srcRow = (const Rpp16u*)((const char*)pSrc + j * nSrcStep);
        hipMemcpy(d_off_in, srcRow, srcDescPtr->w * sizeof(Rpp16u), hipMemcpyDeviceToDevice);
        d_off_in += srcDescPtr->w;
    }
    rppHandle_t handle;
    hipStream_t stream;
    hipStreamCreate(&stream);
    rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);
    status = rppt_remap_gpu(d_input, srcDescPtr, d_output, dstDescPtr, (Rpp32f *)pYMap, (Rpp32f *)pXMap, tableDescPtr, 
							interpolationType, roiTensorPtrSrc, roiTypeSrc, handle);
    hipDeviceSynchronize();
    rppDestroyGPU(handle);
    // Copy output back row by row
    Rpp16u *d_off_out = d_output;
    for(int k = 0; k < dstDescPtr->h; k++) {
        Rpp16u* dstRow = (Rpp16u*)((char*)pDst + k * nDstStep);
        hipMemcpy(dstRow, d_off_out, dstDescPtr->w * sizeof(Rpp16u), hipMemcpyDeviceToDevice);
        d_off_out += dstDescPtr->w;
    }

    hipFree(d_input);
    hipFree(d_output);
    hipHostFree(roiTensorPtrSrc);
    hipHostFree(dstImgSizes);
    return(hipRppStatusTocudaNppStatus(status));
}


NppStatus nppiRemap_16u_C3R_Ctx(const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                 const Npp32f *pXMap, int nXMapStep, const Npp32f *pYMap, int nYMapStep,
                                       Npp16u *pDst, int nDstStep, NppiSize oDstSizeROI, int eInterpolation, NppStreamContext nppStreamCtx)
{
    // Three-channel implementation
    int noOfImages  = 1;
    int ip_channel  = 3;
    RpptDesc srcDesc, dstDesc;
    RpptDescPtr srcDescPtr = &srcDesc;
    RpptDescPtr dstDescPtr = &dstDesc;

    // Use NHWC layout for interleaved C3R
    srcDescPtr->layout      = RpptLayout::NHWC;
    srcDescPtr->dataType    = RpptDataType::U16;
    srcDescPtr->numDims     = 4;
    srcDescPtr->offsetInBytes = 0;
    srcDescPtr->n           = noOfImages;
    srcDescPtr->h           = oSrcSize.height;
    srcDescPtr->w           = oSrcSize.width;
    srcDescPtr->c           = ip_channel;

    dstDescPtr->layout      = RpptLayout::NHWC;
    dstDescPtr->dataType    = RpptDataType::U16;
    dstDescPtr->numDims     = 4;
    dstDescPtr->offsetInBytes = 0;
    dstDescPtr->n           = noOfImages;
    dstDescPtr->h           = oDstSizeROI.height;
    dstDescPtr->w           = oDstSizeROI.width;
    dstDescPtr->c           = ip_channel;

    // Source ROI
    RpptROI *roiTensorPtrSrc;
    hipHostMalloc(&roiTensorPtrSrc, noOfImages * sizeof(RpptROI));
    roiTensorPtrSrc[0].xywhROI.xy.x     = oSrcROI.x;
    roiTensorPtrSrc[0].xywhROI.xy.y     = oSrcROI.y;
    roiTensorPtrSrc[0].xywhROI.roiWidth  = oSrcROI.width;
    roiTensorPtrSrc[0].xywhROI.roiHeight = oSrcROI.height;
    RpptRoiType roiTypeSrc = RpptRoiType::XYWH;

    // Destination patch
    RpptImagePatch *dstImgSizes;
    hipHostMalloc(&dstImgSizes, noOfImages * sizeof(RpptImagePatch));
    dstImgSizes[0].width  = oDstSizeROI.width;
    dstImgSizes[0].height = oDstSizeROI.height;

    // Strides for NHWC: nStride = h*w*c, hStride = w*c, wStride = c, cStride = 1
    srcDescPtr->strides.nStride = srcDescPtr->h * srcDescPtr->w * srcDescPtr->c;
    srcDescPtr->strides.hStride = srcDescPtr->w * srcDescPtr->c;
    srcDescPtr->strides.wStride = srcDescPtr->c;
    srcDescPtr->strides.cStride = 1;

    dstDescPtr->strides.nStride = dstDescPtr->h * dstDescPtr->w * dstDescPtr->c;
    dstDescPtr->strides.hStride = dstDescPtr->w * dstDescPtr->c;
    dstDescPtr->strides.wStride = dstDescPtr->c;
    dstDescPtr->strides.cStride = 1;

    // Table descriptor remains single-channel
    RpptDesc tableDesc;
    RpptDescPtr tableDescPtr = &tableDesc;
    tableDescPtr->layout      = RpptLayout::NHWC;
    tableDescPtr->dataType    = RpptDataType::U16;
    tableDescPtr->numDims     = 4;
    tableDescPtr->offsetInBytes = 0;
    tableDescPtr->n           = noOfImages;
    tableDescPtr->h           = oDstSizeROI.height;
    tableDescPtr->w           = oDstSizeROI.width;
    tableDescPtr->c           = 1;
    tableDescPtr->strides.nStride = tableDescPtr->h * tableDescPtr->w * tableDescPtr->c;
    tableDescPtr->strides.hStride = tableDescPtr->w * tableDescPtr->c;
    tableDescPtr->strides.wStride = tableDescPtr->c;
    tableDescPtr->strides.cStride = 1;

    // Allocate device buffers
    size_t inBufSize  = srcDescPtr->n * srcDescPtr->h * srcDescPtr->w * srcDescPtr->c * sizeof(Rpp16u);
    size_t outBufSize = dstDescPtr->n * dstDescPtr->h * dstDescPtr->w * dstDescPtr->c * sizeof(Rpp16u);
    Rpp16u *d_input, *d_output;
    hipMalloc(&d_input,  inBufSize);
    hipMalloc(&d_output, outBufSize);

    // Choose interpolation
    RpptInterpolationType interpolationType;
    switch(eInterpolation) {
        case NPPI_INTER_NN:      interpolationType = RpptInterpolationType::NEAREST_NEIGHBOR; break;
        case NPPI_INTER_LINEAR:  interpolationType = RpptInterpolationType::BILINEAR;        break;
        case NPPI_INTER_CUBIC:   interpolationType = RpptInterpolationType::BICUBIC;         break;
        case NPPI_INTER_LANCZOS: interpolationType = RpptInterpolationType::LANCZOS;         break;
        default:
            return hipRppStatusTocudaNppStatus(RPP_ERROR_INVALID_ARGUMENTS);
    }

    // Copy host->device row by row
    Rpp16u *d_off_in = d_input;
    for(int y = 0; y < srcDescPtr->h; y++) {
        const Rpp16u* srcRow = (const Rpp16u*)((const char*)pSrc + y * nSrcStep);
        hipMemcpy(d_off_in, srcRow, srcDescPtr->w * srcDescPtr->c * sizeof(Rpp16u), hipMemcpyDeviceToDevice);
        d_off_in += srcDescPtr->w * srcDescPtr->c;
    }

    // Run remap
    rppHandle_t handle;
    hipStream_t stream;
    hipStreamCreate(&stream);
    rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);
    RppStatus status = rppt_remap_gpu(
        d_input, srcDescPtr,
        d_output, dstDescPtr,
        (Rpp32f*)pYMap, (Rpp32f*)pXMap, tableDescPtr,
        interpolationType, roiTensorPtrSrc, roiTypeSrc,
        handle
    );
    hipDeviceSynchronize();
    rppDestroyGPU(handle);

    // Copy device->host
    Rpp16u *d_off_out = d_output;
    for(int y = 0; y < dstDescPtr->h; y++) {
        Rpp16u* dstRow = (Rpp16u*)((char*)pDst + y * nDstStep);
        hipMemcpy(dstRow, d_off_out, dstDescPtr->w * dstDescPtr->c * sizeof(Rpp16u), hipMemcpyDeviceToDevice);
        d_off_out += dstDescPtr->w * dstDescPtr->c;
    }

    hipFree(d_input);
    hipFree(d_output);
    hipHostFree(roiTensorPtrSrc);
    hipHostFree(dstImgSizes);

    return hipRppStatusTocudaNppStatus(status);
}


NppStatus nppiRemap_16s_C1R_Ctx(const Npp16s  * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                 const Npp32f * pXMap, int nXMapStep, const Npp32f * pYMap, int nYMapStep,
                                       Npp16s  * pDst, int nDstStep, NppiSize oDstSizeROI, int eInterpolation, NppStreamContext nppStreamCtx)
{
    //pkd
    int noOfImages = 1;
    int ip_channel = 1;
    RpptDesc srcDesc, dstDesc;
    RpptDescPtr srcDescPtr = &srcDesc;
    RpptDescPtr dstDescPtr = &dstDesc;

    
    srcDescPtr->layout      = RpptLayout::NCHW;
    srcDescPtr->dataType    = RpptDataType::S16;
    srcDescPtr->numDims     = 4;
    srcDescPtr->offsetInBytes = 0;
    srcDescPtr->n           = noOfImages;
    srcDescPtr->c           = ip_channel;
    srcDescPtr->h           = oSrcSize.height;
    srcDescPtr->w           = oSrcSize.width;

    dstDescPtr->layout      = RpptLayout::NCHW;
    dstDescPtr->dataType    = RpptDataType::S16;
    dstDescPtr->numDims     = 4;
    dstDescPtr->offsetInBytes = 0;
    dstDescPtr->n           = noOfImages;
    dstDescPtr->c           = ip_channel;
    dstDescPtr->h           = oDstSizeROI.height;
    dstDescPtr->w           = oDstSizeROI.width;


    // ROI for source
    RpptROI *roiTensorPtrSrc;
    hipHostMalloc(&roiTensorPtrSrc, noOfImages * sizeof(RpptROI));
    roiTensorPtrSrc[0].xywhROI.xy.x     = oSrcROI.x;
    roiTensorPtrSrc[0].xywhROI.xy.y     = oSrcROI.y;
    roiTensorPtrSrc[0].xywhROI.roiWidth  = oSrcROI.width;
    roiTensorPtrSrc[0].xywhROI.roiHeight = oSrcROI.height;
    RpptRoiType roiTypeSrc = RpptRoiType::XYWH;

    // Destination image patch
    RpptImagePatch *dstImgSizes;
    hipHostMalloc(&dstImgSizes, noOfImages * sizeof(RpptImagePatch));
    dstImgSizes[0].width  = oDstSizeROI.width;
    dstImgSizes[0].height = oDstSizeROI.height;

    // Strides for NCHW: nStride = c*h*w, cStride = h*w, hStride = w, wStride = 1
    srcDescPtr->strides.nStride = srcDescPtr->c * srcDescPtr->h * srcDescPtr->w;
    srcDescPtr->strides.cStride = srcDescPtr->h * srcDescPtr->w;
    srcDescPtr->strides.hStride = srcDescPtr->w;
    srcDescPtr->strides.wStride = 1;

    dstDescPtr->strides.nStride = dstDescPtr->c * dstDescPtr->h * dstDescPtr->w;
    dstDescPtr->strides.cStride = dstDescPtr->h * dstDescPtr->w;
    dstDescPtr->strides.hStride = dstDescPtr->w;
    dstDescPtr->strides.wStride = 1;

    RpptDesc tableDesc;
    RpptDescPtr tableDescPtr = &tableDesc;
	
    tableDescPtr->layout      = RpptLayout::NCHW;
    tableDescPtr->dataType    = RpptDataType::S16;
    tableDescPtr->numDims     = 4;
    tableDescPtr->offsetInBytes = 0;
    tableDescPtr->n           = noOfImages;
    tableDescPtr->c           = 1;
    tableDescPtr->h           = oDstSizeROI.height;
    tableDescPtr->w           = oDstSizeROI.width;
    tableDescPtr->strides.nStride = tableDescPtr->c * tableDescPtr->h * tableDescPtr->w;
    tableDescPtr->strides.cStride = tableDescPtr->h * tableDescPtr->w;
    tableDescPtr->strides.hStride = tableDescPtr->w;
    tableDescPtr->strides.wStride = 1;
    
    unsigned long long inBufSize  = (unsigned long long)srcDescPtr->n * srcDescPtr->c * srcDescPtr->h * srcDescPtr->w * sizeof(Rpp16s);
    unsigned long long outBufSize = (unsigned long long)dstDescPtr->n * dstDescPtr->c * dstDescPtr->h * dstDescPtr->w * sizeof(Rpp16s);
    Rpp16s *d_input,*d_output;
	hipMalloc(&d_input,  inBufSize);
    hipMalloc(&d_output, outBufSize);

    // Select interpolation type
	RppStatus status;
    RpptInterpolationType interpolationType = RpptInterpolationType::BILINEAR;
    if(eInterpolation == NPPI_INTER_NN) {
           interpolationType = RpptInterpolationType::NEAREST_NEIGHBOR;
    } else if (eInterpolation == NPPI_INTER_LINEAR) {
           interpolationType = RpptInterpolationType::BILINEAR;
    } else if (eInterpolation == NPPI_INTER_CUBIC) {
           interpolationType = RpptInterpolationType::BICUBIC;
    } else if(eInterpolation == NPPI_INTER_LANCZOS) {
           interpolationType = RpptInterpolationType::LANCZOS;
    } else {
           status = RPP_ERROR_INVALID_ARGUMENTS;
		   return(hipRppStatusTocudaNppStatus(status));
    }


    Rpp16s *d_off_in = d_input;
    for(int j = 0; j < srcDescPtr->h; j++) {
        const Rpp16s* srcRow = (const Rpp16s*)((const char*)pSrc + j * nSrcStep);
        hipMemcpy(d_off_in, srcRow, srcDescPtr->w * sizeof(Rpp16s), hipMemcpyDeviceToDevice);
        d_off_in += srcDescPtr->w;
    }
    rppHandle_t handle;
    hipStream_t stream;
    hipStreamCreate(&stream);
    rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);
    status = rppt_remap_gpu(d_input, srcDescPtr, d_output, dstDescPtr, (Rpp32f *)pYMap, (Rpp32f *)pXMap, tableDescPtr, 
							interpolationType, roiTensorPtrSrc, roiTypeSrc, handle);
    hipDeviceSynchronize();
    rppDestroyGPU(handle);
    // Copy output back row by row
    Rpp16s *d_off_out = d_output;
    for(int k = 0; k < dstDescPtr->h; k++) {
        Rpp16s* dstRow = (Rpp16s*)((char*)pDst + k * nDstStep);
        hipMemcpy(dstRow, d_off_out, dstDescPtr->w * sizeof(Rpp16s), hipMemcpyDeviceToDevice);
        d_off_out += dstDescPtr->w;
    }

    hipFree(d_input);
    hipFree(d_output);
    hipHostFree(roiTensorPtrSrc);
    hipHostFree(dstImgSizes);
    return(hipRppStatusTocudaNppStatus(status));
}


NppStatus nppiRemap_16s_C3R_Ctx(const Npp16s  * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                     const Npp32f * pXMap, int nXMapStep, const Npp32f * pYMap, int nYMapStep,
                           Npp16s  * pDst, int nDstStep, NppiSize oDstSizeROI, int eInterpolation, NppStreamContext nppStreamCtx)
{
    // Three-channel implementation
    int noOfImages  = 1;
    int ip_channel  = 3;
    RpptDesc srcDesc, dstDesc;
    RpptDescPtr srcDescPtr = &srcDesc;
    RpptDescPtr dstDescPtr = &dstDesc;

    // Use NHWC layout for interleaved C3R
    srcDescPtr->layout      = RpptLayout::NHWC;
    srcDescPtr->dataType    = RpptDataType::S16;
    srcDescPtr->numDims     = 4;
    srcDescPtr->offsetInBytes = 0;
    srcDescPtr->n           = noOfImages;
    srcDescPtr->h           = oSrcSize.height;
    srcDescPtr->w           = oSrcSize.width;
    srcDescPtr->c           = ip_channel;

    dstDescPtr->layout      = RpptLayout::NHWC;
    dstDescPtr->dataType    = RpptDataType::S16;
    dstDescPtr->numDims     = 4;
    dstDescPtr->offsetInBytes = 0;
    dstDescPtr->n           = noOfImages;
    dstDescPtr->h           = oDstSizeROI.height;
    dstDescPtr->w           = oDstSizeROI.width;
    dstDescPtr->c           = ip_channel;

    // Source ROI
    RpptROI *roiTensorPtrSrc;
    hipHostMalloc(&roiTensorPtrSrc, noOfImages * sizeof(RpptROI));
    roiTensorPtrSrc[0].xywhROI.xy.x     = oSrcROI.x;
    roiTensorPtrSrc[0].xywhROI.xy.y     = oSrcROI.y;
    roiTensorPtrSrc[0].xywhROI.roiWidth  = oSrcROI.width;
    roiTensorPtrSrc[0].xywhROI.roiHeight = oSrcROI.height;
    RpptRoiType roiTypeSrc = RpptRoiType::XYWH;

    // Destination patch
    RpptImagePatch *dstImgSizes;
    hipHostMalloc(&dstImgSizes, noOfImages * sizeof(RpptImagePatch));
    dstImgSizes[0].width  = oDstSizeROI.width;
    dstImgSizes[0].height = oDstSizeROI.height;

    // Strides for NHWC: nStride = h*w*c, hStride = w*c, wStride = c, cStride = 1
    srcDescPtr->strides.nStride = srcDescPtr->h * srcDescPtr->w * srcDescPtr->c;
    srcDescPtr->strides.hStride = srcDescPtr->w * srcDescPtr->c;
    srcDescPtr->strides.wStride = srcDescPtr->c;
    srcDescPtr->strides.cStride = 1;

    dstDescPtr->strides.nStride = dstDescPtr->h * dstDescPtr->w * dstDescPtr->c;
    dstDescPtr->strides.hStride = dstDescPtr->w * dstDescPtr->c;
    dstDescPtr->strides.wStride = dstDescPtr->c;
    dstDescPtr->strides.cStride = 1;

    // Table descriptor remains single-channel
    RpptDesc tableDesc;
    RpptDescPtr tableDescPtr = &tableDesc;
    tableDescPtr->layout      = RpptLayout::NHWC;
    tableDescPtr->dataType    = RpptDataType::S16;
    tableDescPtr->numDims     = 4;
    tableDescPtr->offsetInBytes = 0;
    tableDescPtr->n           = noOfImages;
    tableDescPtr->h           = oDstSizeROI.height;
    tableDescPtr->w           = oDstSizeROI.width;
    tableDescPtr->c           = 1;
    tableDescPtr->strides.nStride = tableDescPtr->h * tableDescPtr->w * tableDescPtr->c;
    tableDescPtr->strides.hStride = tableDescPtr->w * tableDescPtr->c;
    tableDescPtr->strides.wStride = tableDescPtr->c;
    tableDescPtr->strides.cStride = 1;

    // Allocate device buffers
    size_t inBufSize  = srcDescPtr->n * srcDescPtr->h * srcDescPtr->w * srcDescPtr->c * sizeof(Rpp16s);
    size_t outBufSize = dstDescPtr->n * dstDescPtr->h * dstDescPtr->w * dstDescPtr->c * sizeof(Rpp16s);
    Rpp16s *d_input, *d_output;
    hipMalloc(&d_input,  inBufSize);
    hipMalloc(&d_output, outBufSize);

    // Choose interpolation
    RpptInterpolationType interpolationType;
    switch(eInterpolation) {
        case NPPI_INTER_NN:      interpolationType = RpptInterpolationType::NEAREST_NEIGHBOR; break;
        case NPPI_INTER_LINEAR:  interpolationType = RpptInterpolationType::BILINEAR;        break;
        case NPPI_INTER_CUBIC:   interpolationType = RpptInterpolationType::BICUBIC;         break;
        case NPPI_INTER_LANCZOS: interpolationType = RpptInterpolationType::LANCZOS;         break;
        default:
            return hipRppStatusTocudaNppStatus(RPP_ERROR_INVALID_ARGUMENTS);
    }

    // Copy host->device row by row
    Rpp16s *d_off_in = d_input;
    for(int y = 0; y < srcDescPtr->h; y++) {
        const Rpp16s* srcRow = (const Rpp16s*)((const char*)pSrc + y * nSrcStep);
        hipMemcpy(d_off_in, srcRow, srcDescPtr->w * srcDescPtr->c * sizeof(Rpp16s), hipMemcpyDeviceToDevice);
        d_off_in += srcDescPtr->w * srcDescPtr->c;
    }

    // Run remap
    rppHandle_t handle;
    hipStream_t stream;
    hipStreamCreate(&stream);
    rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);
    RppStatus status = rppt_remap_gpu(
        d_input, srcDescPtr,
        d_output, dstDescPtr,
        (Rpp32f*)pYMap, (Rpp32f*)pXMap, tableDescPtr,
        interpolationType, roiTensorPtrSrc, roiTypeSrc,
        handle
    );
    hipDeviceSynchronize();
    rppDestroyGPU(handle);

    // Copy device->host
    Rpp16s *d_off_out = d_output;
    for(int y = 0; y < dstDescPtr->h; y++) {
        Rpp16s* dstRow = (Rpp16s*)((char*)pDst + y * nDstStep);
        hipMemcpy(dstRow, d_off_out, dstDescPtr->w * dstDescPtr->c * sizeof(Rpp16s), hipMemcpyDeviceToDevice);
        d_off_out += dstDescPtr->w * dstDescPtr->c;
    }

    hipFree(d_input);
    hipFree(d_output);
    hipHostFree(roiTensorPtrSrc);
    hipHostFree(dstImgSizes);

    return hipRppStatusTocudaNppStatus(status);
}


NppStatus nppiRemap_32f_C1R_Ctx(const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                 const Npp32f *pXMap, int nXMapStep, const Npp32f *pYMap, int nYMapStep,
                                       Npp32f *pDst, int nDstStep, NppiSize oDstSizeROI, int eInterpolation, NppStreamContext nppStreamCtx)
{
    //pkd
    int noOfImages = 1;
    int ip_channel = 1;
    RpptDesc srcDesc, dstDesc;
    RpptDescPtr srcDescPtr = &srcDesc;
    RpptDescPtr dstDescPtr = &dstDesc;

    
    srcDescPtr->layout      = RpptLayout::NCHW;
    srcDescPtr->dataType    = RpptDataType::F32;
    srcDescPtr->numDims     = 4;
    srcDescPtr->offsetInBytes = 0;
    srcDescPtr->n           = noOfImages;
    srcDescPtr->c           = ip_channel;
    srcDescPtr->h           = oSrcSize.height;
    srcDescPtr->w           = oSrcSize.width;

    dstDescPtr->layout      = RpptLayout::NCHW;
    dstDescPtr->dataType    = RpptDataType::F32;
    dstDescPtr->numDims     = 4;
    dstDescPtr->offsetInBytes = 0;
    dstDescPtr->n           = noOfImages;
    dstDescPtr->c           = ip_channel;
    dstDescPtr->h           = oDstSizeROI.height;
    dstDescPtr->w           = oDstSizeROI.width;


    // ROI for source
    RpptROI *roiTensorPtrSrc;
    hipHostMalloc(&roiTensorPtrSrc, noOfImages * sizeof(RpptROI));
    roiTensorPtrSrc[0].xywhROI.xy.x     = oSrcROI.x;
    roiTensorPtrSrc[0].xywhROI.xy.y     = oSrcROI.y;
    roiTensorPtrSrc[0].xywhROI.roiWidth  = oSrcROI.width;
    roiTensorPtrSrc[0].xywhROI.roiHeight = oSrcROI.height;
    RpptRoiType roiTypeSrc = RpptRoiType::XYWH;

    // Destination image patch
    RpptImagePatch *dstImgSizes;
    hipHostMalloc(&dstImgSizes, noOfImages * sizeof(RpptImagePatch));
    dstImgSizes[0].width  = oDstSizeROI.width;
    dstImgSizes[0].height = oDstSizeROI.height;

    // Strides for NCHW: nStride = c*h*w, cStride = h*w, hStride = w, wStride = 1
    srcDescPtr->strides.nStride = srcDescPtr->c * srcDescPtr->h * srcDescPtr->w;
    srcDescPtr->strides.cStride = srcDescPtr->h * srcDescPtr->w;
    srcDescPtr->strides.hStride = srcDescPtr->w;
    srcDescPtr->strides.wStride = 1;

    dstDescPtr->strides.nStride = dstDescPtr->c * dstDescPtr->h * dstDescPtr->w;
    dstDescPtr->strides.cStride = dstDescPtr->h * dstDescPtr->w;
    dstDescPtr->strides.hStride = dstDescPtr->w;
    dstDescPtr->strides.wStride = 1;

    RpptDesc tableDesc;
    RpptDescPtr tableDescPtr = &tableDesc;
	
    tableDescPtr->layout      = RpptLayout::NCHW;
    tableDescPtr->dataType    = RpptDataType::F32;
    tableDescPtr->numDims     = 4;
    tableDescPtr->offsetInBytes = 0;
    tableDescPtr->n           = noOfImages;
    tableDescPtr->c           = 1;
    tableDescPtr->h           = oDstSizeROI.height;
    tableDescPtr->w           = oDstSizeROI.width;
    tableDescPtr->strides.nStride = tableDescPtr->c * tableDescPtr->h * tableDescPtr->w;
    tableDescPtr->strides.cStride = tableDescPtr->h * tableDescPtr->w;
    tableDescPtr->strides.hStride = tableDescPtr->w;
    tableDescPtr->strides.wStride = 1;
    
    unsigned long long inBufSize  = (unsigned long long)srcDescPtr->n * srcDescPtr->c * srcDescPtr->h * srcDescPtr->w * sizeof(Rpp32f);
    unsigned long long outBufSize = (unsigned long long)dstDescPtr->n * dstDescPtr->c * dstDescPtr->h * dstDescPtr->w * sizeof(Rpp32f);
    Rpp32f *d_input,*d_output;
	hipMalloc(&d_input,  inBufSize);
    hipMalloc(&d_output, outBufSize);

    // Select interpolation type
	RppStatus status;
    RpptInterpolationType interpolationType = RpptInterpolationType::BILINEAR;
    if(eInterpolation == NPPI_INTER_NN) {
           interpolationType = RpptInterpolationType::NEAREST_NEIGHBOR;
    } else if (eInterpolation == NPPI_INTER_LINEAR) {
           interpolationType = RpptInterpolationType::BILINEAR;
    } else if (eInterpolation == NPPI_INTER_CUBIC) {
           interpolationType = RpptInterpolationType::BICUBIC;
    } else if(eInterpolation == NPPI_INTER_LANCZOS) {
           interpolationType = RpptInterpolationType::LANCZOS;
    } else {
           status = RPP_ERROR_INVALID_ARGUMENTS;
		   return(hipRppStatusTocudaNppStatus(status));
    }


    Rpp32f *d_off_in = d_input;
    for(int j = 0; j < srcDescPtr->h; j++) {
        const Rpp32f* srcRow = (const Rpp32f*)((const char*)pSrc + j * nSrcStep);
        hipMemcpy(d_off_in, srcRow, srcDescPtr->w * sizeof(Rpp32f), hipMemcpyDeviceToDevice);
        d_off_in += srcDescPtr->w;
    }
    rppHandle_t handle;
    hipStream_t stream;
    hipStreamCreate(&stream);
    rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);
    status = rppt_remap_gpu(d_input, srcDescPtr, d_output, dstDescPtr, (Rpp32f *)pYMap, (Rpp32f *)pXMap, tableDescPtr, 
							interpolationType, roiTensorPtrSrc, roiTypeSrc, handle);
    hipDeviceSynchronize();
    rppDestroyGPU(handle);
    // Copy output back row by row
    Rpp32f *d_off_out = d_output;
    for(int k = 0; k < dstDescPtr->h; k++) {
        Rpp32f* dstRow = (Rpp32f*)((char*)pDst + k * nDstStep);
        hipMemcpy(dstRow, d_off_out, dstDescPtr->w * sizeof(Rpp32f), hipMemcpyDeviceToDevice);
        d_off_out += dstDescPtr->w;
    }

    hipFree(d_input);
    hipFree(d_output);
    hipHostFree(roiTensorPtrSrc);
    hipHostFree(dstImgSizes);
    return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiRemap_32f_C3R_Ctx(const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                 const Npp32f *pXMap, int nXMapStep, const Npp32f *pYMap, int nYMapStep,
                                       Npp32f *pDst, int nDstStep, NppiSize oDstSizeROI, int eInterpolation, NppStreamContext nppStreamCtx)
{
    // Three-channel implementation
    int noOfImages  = 1;
    int ip_channel  = 3;
    RpptDesc srcDesc, dstDesc;
    RpptDescPtr srcDescPtr = &srcDesc;
    RpptDescPtr dstDescPtr = &dstDesc;

    // Use NHWC layout for interleaved C3R
    srcDescPtr->layout      = RpptLayout::NHWC;
    srcDescPtr->dataType    = RpptDataType::F32;
    srcDescPtr->numDims     = 4;
    srcDescPtr->offsetInBytes = 0;
    srcDescPtr->n           = noOfImages;
    srcDescPtr->h           = oSrcSize.height;
    srcDescPtr->w           = oSrcSize.width;
    srcDescPtr->c           = ip_channel;

    dstDescPtr->layout      = RpptLayout::NHWC;
    dstDescPtr->dataType    = RpptDataType::F32;
    dstDescPtr->numDims     = 4;
    dstDescPtr->offsetInBytes = 0;
    dstDescPtr->n           = noOfImages;
    dstDescPtr->h           = oDstSizeROI.height;
    dstDescPtr->w           = oDstSizeROI.width;
    dstDescPtr->c           = ip_channel;

    // Source ROI
    RpptROI *roiTensorPtrSrc;
    hipHostMalloc(&roiTensorPtrSrc, noOfImages * sizeof(RpptROI));
    roiTensorPtrSrc[0].xywhROI.xy.x     = oSrcROI.x;
    roiTensorPtrSrc[0].xywhROI.xy.y     = oSrcROI.y;
    roiTensorPtrSrc[0].xywhROI.roiWidth  = oSrcROI.width;
    roiTensorPtrSrc[0].xywhROI.roiHeight = oSrcROI.height;
    RpptRoiType roiTypeSrc = RpptRoiType::XYWH;

    // Destination patch
    RpptImagePatch *dstImgSizes;
    hipHostMalloc(&dstImgSizes, noOfImages * sizeof(RpptImagePatch));
    dstImgSizes[0].width  = oDstSizeROI.width;
    dstImgSizes[0].height = oDstSizeROI.height;

    // Strides for NHWC: nStride = h*w*c, hStride = w*c, wStride = c, cStride = 1
    srcDescPtr->strides.nStride = srcDescPtr->h * srcDescPtr->w * srcDescPtr->c;
    srcDescPtr->strides.hStride = srcDescPtr->w * srcDescPtr->c;
    srcDescPtr->strides.wStride = srcDescPtr->c;
    srcDescPtr->strides.cStride = 1;

    dstDescPtr->strides.nStride = dstDescPtr->h * dstDescPtr->w * dstDescPtr->c;
    dstDescPtr->strides.hStride = dstDescPtr->w * dstDescPtr->c;
    dstDescPtr->strides.wStride = dstDescPtr->c;
    dstDescPtr->strides.cStride = 1;

    // Table descriptor remains single-channel
    RpptDesc tableDesc;
    RpptDescPtr tableDescPtr = &tableDesc;
    tableDescPtr->layout      = RpptLayout::NHWC;
    tableDescPtr->dataType    = RpptDataType::F32;
    tableDescPtr->numDims     = 4;
    tableDescPtr->offsetInBytes = 0;
    tableDescPtr->n           = noOfImages;
    tableDescPtr->h           = oDstSizeROI.height;
    tableDescPtr->w           = oDstSizeROI.width;
    tableDescPtr->c           = 1;
    tableDescPtr->strides.nStride = tableDescPtr->h * tableDescPtr->w * tableDescPtr->c;
    tableDescPtr->strides.hStride = tableDescPtr->w * tableDescPtr->c;
    tableDescPtr->strides.wStride = tableDescPtr->c;
    tableDescPtr->strides.cStride = 1;

    // Allocate device buffers
    size_t inBufSize  = srcDescPtr->n * srcDescPtr->h * srcDescPtr->w * srcDescPtr->c * sizeof(Rpp32f);
    size_t outBufSize = dstDescPtr->n * dstDescPtr->h * dstDescPtr->w * dstDescPtr->c * sizeof(Rpp32f);
    Rpp32f *d_input, *d_output;
    hipMalloc(&d_input,  inBufSize);
    hipMalloc(&d_output, outBufSize);

    // Choose interpolation
    RpptInterpolationType interpolationType;
    switch(eInterpolation) {
        case NPPI_INTER_NN:      interpolationType = RpptInterpolationType::NEAREST_NEIGHBOR; break;
        case NPPI_INTER_LINEAR:  interpolationType = RpptInterpolationType::BILINEAR;        break;
        case NPPI_INTER_CUBIC:   interpolationType = RpptInterpolationType::BICUBIC;         break;
        case NPPI_INTER_LANCZOS: interpolationType = RpptInterpolationType::LANCZOS;         break;
        default:
            return hipRppStatusTocudaNppStatus(RPP_ERROR_INVALID_ARGUMENTS);
    }

    // Copy host->device row by row
    Rpp32f *d_off_in = d_input;
    for(int y = 0; y < srcDescPtr->h; y++) {
        const Rpp32f* srcRow = (const Rpp32f*)((const char*)pSrc + y * nSrcStep);
        hipMemcpy(d_off_in, srcRow, srcDescPtr->w * srcDescPtr->c * sizeof(Rpp32f), hipMemcpyDeviceToDevice);
        d_off_in += srcDescPtr->w * srcDescPtr->c;
    }

    // Run remap
    rppHandle_t handle;
    hipStream_t stream;
    hipStreamCreate(&stream);
    rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);
    RppStatus status = rppt_remap_gpu(
        d_input, srcDescPtr,
        d_output, dstDescPtr,
        (Rpp32f*)pYMap, (Rpp32f*)pXMap, tableDescPtr,
        interpolationType, roiTensorPtrSrc, roiTypeSrc,
        handle
    );
    hipDeviceSynchronize();
    rppDestroyGPU(handle);

    // Copy device->host
    Rpp32f *d_off_out = d_output;
    for(int y = 0; y < dstDescPtr->h; y++) {
        Rpp32f* dstRow = (Rpp32f*)((char*)pDst + y * nDstStep);
        hipMemcpy(dstRow, d_off_out, dstDescPtr->w * dstDescPtr->c * sizeof(Rpp32f), hipMemcpyDeviceToDevice);
        d_off_out += dstDescPtr->w * dstDescPtr->c;
    }

    hipFree(d_input);
    hipFree(d_output);
    hipHostFree(roiTensorPtrSrc);
    hipHostFree(dstImgSizes);

    return hipRppStatusTocudaNppStatus(status);
}


__global__ void d2f_cast_kernel(const double* in, float* out, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N) out[idx] = static_cast<float>(in[idx]);
}

NppStatus nppiRemap_64f_C1R_Ctx(const Npp64f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                 const Npp64f *pXMap, int nXMapStep, const Npp64f *pYMap, int nYMapStep,
                                       Npp64f *pDst, int nDstStep, NppiSize oDstSizeROI, int eInterpolation, NppStreamContext nppStreamCtx)
{
    //pkd
    int noOfImages = 1;
    int ip_channel = 1;
    RpptDesc srcDesc, dstDesc;
    RpptDescPtr srcDescPtr = &srcDesc;
    RpptDescPtr dstDescPtr = &dstDesc;

    
    srcDescPtr->layout      = RpptLayout::NCHW;
    srcDescPtr->dataType    = RpptDataType::F64;
    srcDescPtr->numDims     = 4;
    srcDescPtr->offsetInBytes = 0;
    srcDescPtr->n           = noOfImages;
    srcDescPtr->c           = ip_channel;
    srcDescPtr->h           = oSrcSize.height;
    srcDescPtr->w           = oSrcSize.width;

    dstDescPtr->layout      = RpptLayout::NCHW;
    dstDescPtr->dataType    = RpptDataType::F64;
    dstDescPtr->numDims     = 4;
    dstDescPtr->offsetInBytes = 0;
    dstDescPtr->n           = noOfImages;
    dstDescPtr->c           = ip_channel;
    dstDescPtr->h           = oDstSizeROI.height;
    dstDescPtr->w           = oDstSizeROI.width;


    // ROI for source
    RpptROI *roiTensorPtrSrc;
    hipHostMalloc(&roiTensorPtrSrc, noOfImages * sizeof(RpptROI));
    roiTensorPtrSrc[0].xywhROI.xy.x     = oSrcROI.x;
    roiTensorPtrSrc[0].xywhROI.xy.y     = oSrcROI.y;
    roiTensorPtrSrc[0].xywhROI.roiWidth  = oSrcROI.width;
    roiTensorPtrSrc[0].xywhROI.roiHeight = oSrcROI.height;
    RpptRoiType roiTypeSrc = RpptRoiType::XYWH;

    // Destination image patch
    RpptImagePatch *dstImgSizes;
    hipHostMalloc(&dstImgSizes, noOfImages * sizeof(RpptImagePatch));
    dstImgSizes[0].width  = oDstSizeROI.width;
    dstImgSizes[0].height = oDstSizeROI.height;

    // Strides for NCHW: nStride = c*h*w, cStride = h*w, hStride = w, wStride = 1
    srcDescPtr->strides.nStride = srcDescPtr->c * srcDescPtr->h * srcDescPtr->w;
    srcDescPtr->strides.cStride = srcDescPtr->h * srcDescPtr->w;
    srcDescPtr->strides.hStride = srcDescPtr->w;
    srcDescPtr->strides.wStride = 1;

    dstDescPtr->strides.nStride = dstDescPtr->c * dstDescPtr->h * dstDescPtr->w;
    dstDescPtr->strides.cStride = dstDescPtr->h * dstDescPtr->w;
    dstDescPtr->strides.hStride = dstDescPtr->w;
    dstDescPtr->strides.wStride = 1;

    RpptDesc tableDesc;
    RpptDescPtr tableDescPtr = &tableDesc;
	
    tableDescPtr->layout      = RpptLayout::NCHW;
    tableDescPtr->dataType    = RpptDataType::F64;
    tableDescPtr->numDims     = 4;
    tableDescPtr->offsetInBytes = 0;
    tableDescPtr->n           = noOfImages;
    tableDescPtr->c           = 1;
    tableDescPtr->h           = oDstSizeROI.height;
    tableDescPtr->w           = oDstSizeROI.width;
    tableDescPtr->strides.nStride = tableDescPtr->c * tableDescPtr->h * tableDescPtr->w;
    tableDescPtr->strides.cStride = tableDescPtr->h * tableDescPtr->w;
    tableDescPtr->strides.hStride = tableDescPtr->w;
    tableDescPtr->strides.wStride = 1;
    
    unsigned long long inBufSize  = (unsigned long long)srcDescPtr->n * srcDescPtr->c * srcDescPtr->h * srcDescPtr->w * sizeof(Rpp64f);
    unsigned long long outBufSize = (unsigned long long)dstDescPtr->n * dstDescPtr->c * dstDescPtr->h * dstDescPtr->w * sizeof(Rpp64f);
    Rpp64f *d_input,*d_output;
	hipMalloc(&d_input,  inBufSize);
    hipMalloc(&d_output, outBufSize);

    // Select interpolation type
	RppStatus status;
    RpptInterpolationType interpolationType = RpptInterpolationType::BILINEAR;
    if(eInterpolation == NPPI_INTER_NN) {
           interpolationType = RpptInterpolationType::NEAREST_NEIGHBOR;
    } else if (eInterpolation == NPPI_INTER_LINEAR) {
           interpolationType = RpptInterpolationType::BILINEAR;
    } else if (eInterpolation == NPPI_INTER_CUBIC) {
           interpolationType = RpptInterpolationType::BICUBIC;
    } else if(eInterpolation == NPPI_INTER_LANCZOS) {
           interpolationType = RpptInterpolationType::LANCZOS;
    } else {
           status = RPP_ERROR_INVALID_ARGUMENTS;
		   return(hipRppStatusTocudaNppStatus(status));
    }


    Rpp64f *d_off_in = d_input;
    for(int j = 0; j < srcDescPtr->h; j++) {
        const Rpp64f* srcRow = (const Rpp64f*)((const char*)pSrc + j * nSrcStep);
        hipMemcpy(d_off_in, srcRow, srcDescPtr->w * sizeof(Rpp64f), hipMemcpyDeviceToDevice);
        d_off_in += srcDescPtr->w;
    }


    int N = tableDescPtr->h * tableDescPtr->w;
    Rpp32f *dXMapF, *dYMapF;
    hipMalloc(&dXMapF, N * sizeof(Rpp32f));
    hipMalloc(&dYMapF, N * sizeof(Rpp32f));

    // launch conversion
    int block = 256, grid = (N + block - 1) / block;
    d2f_cast_kernel<<<grid, block>>>((const double*)pXMap, dXMapF, N);
    d2f_cast_kernel<<<grid, block>>>((const double*)pYMap, dYMapF, N);

    rppHandle_t handle;
    hipStream_t stream;
    hipStreamCreate(&stream);
    rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);
    status = rppt_remap_gpu(d_input, srcDescPtr, d_output, dstDescPtr, dYMapF, dXMapF, tableDescPtr, 
							interpolationType, roiTensorPtrSrc, roiTypeSrc, handle);
    hipDeviceSynchronize();
    rppDestroyGPU(handle);
    // Copy output back row by row
    Rpp64f *d_off_out = d_output;
    for(int k = 0; k < dstDescPtr->h; k++) {
        Rpp64f* dstRow = (Rpp64f*)((char*)pDst + k * nDstStep);
        hipMemcpy(dstRow, d_off_out, dstDescPtr->w * sizeof(Rpp64f), hipMemcpyDeviceToDevice);
        d_off_out += dstDescPtr->w;
    }

    hipFree(d_input);
    hipFree(d_output);
    hipHostFree(roiTensorPtrSrc);
    hipHostFree(dstImgSizes);
    hipFree(dXMapF); hipFree(dYMapF);
    return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiRemap_64f_C3R_Ctx(const Npp64f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                 const Npp64f *pXMap, int nXMapStep, const Npp64f *pYMap, int nYMapStep,
                                       Npp64f *pDst, int nDstStep, NppiSize oDstSizeROI, int eInterpolation, NppStreamContext nppStreamCtx)
{
    // Three-channel implementation
    int noOfImages  = 1;
    int ip_channel  = 3;
    RpptDesc srcDesc, dstDesc;
    RpptDescPtr srcDescPtr = &srcDesc;
    RpptDescPtr dstDescPtr = &dstDesc;

    // Use NHWC layout for interleaved C3R
    srcDescPtr->layout      = RpptLayout::NHWC;
    srcDescPtr->dataType    = RpptDataType::F64;
    srcDescPtr->numDims     = 4;
    srcDescPtr->offsetInBytes = 0;
    srcDescPtr->n           = noOfImages;
    srcDescPtr->h           = oSrcSize.height;
    srcDescPtr->w           = oSrcSize.width;
    srcDescPtr->c           = ip_channel;

    dstDescPtr->layout      = RpptLayout::NHWC;
    dstDescPtr->dataType    = RpptDataType::F64;
    dstDescPtr->numDims     = 4;
    dstDescPtr->offsetInBytes = 0;
    dstDescPtr->n           = noOfImages;
    dstDescPtr->h           = oDstSizeROI.height;
    dstDescPtr->w           = oDstSizeROI.width;
    dstDescPtr->c           = ip_channel;

    // Source ROI
    RpptROI *roiTensorPtrSrc;
    hipHostMalloc(&roiTensorPtrSrc, noOfImages * sizeof(RpptROI));
    roiTensorPtrSrc[0].xywhROI.xy.x     = oSrcROI.x;
    roiTensorPtrSrc[0].xywhROI.xy.y     = oSrcROI.y;
    roiTensorPtrSrc[0].xywhROI.roiWidth  = oSrcROI.width;
    roiTensorPtrSrc[0].xywhROI.roiHeight = oSrcROI.height;
    RpptRoiType roiTypeSrc = RpptRoiType::XYWH;

    // Destination patch
    RpptImagePatch *dstImgSizes;
    hipHostMalloc(&dstImgSizes, noOfImages * sizeof(RpptImagePatch));
    dstImgSizes[0].width  = oDstSizeROI.width;
    dstImgSizes[0].height = oDstSizeROI.height;

    // Strides for NHWC: nStride = h*w*c, hStride = w*c, wStride = c, cStride = 1
    srcDescPtr->strides.nStride = srcDescPtr->h * srcDescPtr->w * srcDescPtr->c;
    srcDescPtr->strides.hStride = srcDescPtr->w * srcDescPtr->c;
    srcDescPtr->strides.wStride = srcDescPtr->c;
    srcDescPtr->strides.cStride = 1;

    dstDescPtr->strides.nStride = dstDescPtr->h * dstDescPtr->w * dstDescPtr->c;
    dstDescPtr->strides.hStride = dstDescPtr->w * dstDescPtr->c;
    dstDescPtr->strides.wStride = dstDescPtr->c;
    dstDescPtr->strides.cStride = 1;

    // Table descriptor remains single-channel
    RpptDesc tableDesc;
    RpptDescPtr tableDescPtr = &tableDesc;
    tableDescPtr->layout      = RpptLayout::NHWC;
    tableDescPtr->dataType    = RpptDataType::F64;
    tableDescPtr->numDims     = 4;
    tableDescPtr->offsetInBytes = 0;
    tableDescPtr->n           = noOfImages;
    tableDescPtr->h           = oDstSizeROI.height;
    tableDescPtr->w           = oDstSizeROI.width;
    tableDescPtr->c           = 1;
    tableDescPtr->strides.nStride = tableDescPtr->h * tableDescPtr->w * tableDescPtr->c;
    tableDescPtr->strides.hStride = tableDescPtr->w * tableDescPtr->c;
    tableDescPtr->strides.wStride = tableDescPtr->c;
    tableDescPtr->strides.cStride = 1;

    // Allocate device buffers
    size_t inBufSize  = srcDescPtr->n * srcDescPtr->h * srcDescPtr->w * srcDescPtr->c * sizeof(Rpp64f);
    size_t outBufSize = dstDescPtr->n * dstDescPtr->h * dstDescPtr->w * dstDescPtr->c * sizeof(Rpp64f);
    Rpp64f *d_input, *d_output;
    hipMalloc(&d_input,  inBufSize);
    hipMalloc(&d_output, outBufSize);

    // Choose interpolation
    RpptInterpolationType interpolationType;
    switch(eInterpolation) {
        case NPPI_INTER_NN:      interpolationType = RpptInterpolationType::NEAREST_NEIGHBOR; break;
        case NPPI_INTER_LINEAR:  interpolationType = RpptInterpolationType::BILINEAR;        break;
        case NPPI_INTER_CUBIC:   interpolationType = RpptInterpolationType::BICUBIC;         break;
        case NPPI_INTER_LANCZOS: interpolationType = RpptInterpolationType::LANCZOS;         break;
        default:
            return hipRppStatusTocudaNppStatus(RPP_ERROR_INVALID_ARGUMENTS);
    }

    // Copy host->device row by row
    Rpp64f *d_off_in = d_input;
    for(int y = 0; y < srcDescPtr->h; y++) {
        const Rpp64f* srcRow = (const Rpp64f*)((const char*)pSrc + y * nSrcStep);
        hipMemcpy(d_off_in, srcRow, srcDescPtr->w * srcDescPtr->c * sizeof(Rpp64f), hipMemcpyDeviceToDevice);
        d_off_in += srcDescPtr->w * srcDescPtr->c;
    }

    int N = tableDescPtr->h * tableDescPtr->w;
    Rpp32f *dXMapF, *dYMapF;
    hipMalloc(&dXMapF, N * sizeof(Rpp32f));
    hipMalloc(&dYMapF, N * sizeof(Rpp32f));

    // launch conversion
    int block = 256, grid = (N + block - 1) / block;
    d2f_cast_kernel<<<grid, block>>>((const double*)pXMap, dXMapF, N);
    d2f_cast_kernel<<<grid, block>>>((const double*)pYMap, dYMapF, N);

    // Run remap
    rppHandle_t handle;
    hipStream_t stream;
    hipStreamCreate(&stream);
    rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);
    RppStatus status = rppt_remap_gpu(
        d_input, srcDescPtr,
        d_output, dstDescPtr,
        dYMapF, dXMapF, tableDescPtr,
        interpolationType, roiTensorPtrSrc, roiTypeSrc,
        handle
    );
    hipDeviceSynchronize();
    rppDestroyGPU(handle);

    // Copy device->host
    Rpp64f *d_off_out = d_output;
    for(int y = 0; y < dstDescPtr->h; y++) {
        Rpp64f* dstRow = (Rpp64f*)((char*)pDst + y * nDstStep);
        hipMemcpy(dstRow, d_off_out, dstDescPtr->w * dstDescPtr->c * sizeof(Rpp64f), hipMemcpyDeviceToDevice);
        d_off_out += dstDescPtr->w * dstDescPtr->c;
    }

    hipFree(d_input);
    hipFree(d_output);
    hipFree(dXMapF); hipFree(dYMapF);
    hipHostFree(roiTensorPtrSrc);
    hipHostFree(dstImgSizes);

    return hipRppStatusTocudaNppStatus(status);
}

#endif // GPU_SUPPORT
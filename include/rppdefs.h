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

#ifndef RPPDEFS_H
#define RPPDEFS_H

/*! \file
 * \brief RPP common HOST/GPU typedef, enum and structure definitions.
 * \defgroup group_rppdefs RPP common definitions
 * \brief RPP definitions for all common HOST/GPU typedefs, enums and structures.
 */

#include <stddef.h>
#include <cmath>
#ifdef HIP_COMPILE
    #include <hip/hip_fp16.h>
#endif // HIP_COMPILE
#include <half/half.hpp>
using halfhpp = half_float::half;
typedef halfhpp Rpp16f;

#ifdef OCL_COMPILE
#include <CL/cl.h>
#endif

#if _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#include <smmintrin.h>
#include <immintrin.h>
#endif
#include<vector>

/*! \brief 8 bit unsigned char minimum \ingroup group_rppdefs \page subpage_rpp */
#define RPP_MIN_8U      ( 0 )
/*! \brief 8 bit unsigned char maximum \ingroup group_rppdefs \page subpage_rppi */
#define RPP_MAX_8U      ( 255 )
/*! \brief RPP maximum dimensions in tensor \ingroup group_rppdefs \page subpage_rppt */
#define RPPT_MAX_DIMS   ( 5 )
/*! \brief RPP maximum channels in audio tensor \ingroup group_rppdefs \page subpage_rppt */
#define RPPT_MAX_AUDIO_CHANNELS   ( 16 )

#define CHECK_RETURN_STATUS(x) do { \
  int retval = (x); \
  if (retval != 0) { \
    fprintf(stderr, "Runtime error: %s returned %d at %s:%d", #x, retval, __FILE__, __LINE__); \
    exit(-1); \
  } \
} while (0)

#ifdef HIP_COMPILE
#include <hip/hip_runtime.h>
#define RPP_HOST_DEVICE __host__ __device__
#else
#define RPP_HOST_DEVICE
#endif

const float ONE_OVER_6                      = 1.0f / 6;
const float ONE_OVER_3                      = 1.0f / 3;
const float ONE_OVER_255                    = 1.0f / 255;
const uint MMS_MAX_SCRATCH_MEMORY           = 115293120; // maximum scratch memory size (in number of floats) needed for MMS buffer in RNNT training
const uint SPECTROGRAM_MAX_SCRATCH_MEMORY   = 372877312; // maximum scratch memory size (in number of floats) needed for spectrogram HIP kernel in RNNT training

/******************** RPP typedefs ********************/

/*! \brief 8 bit unsigned char \ingroup group_rppdefs */
typedef unsigned char       Rpp8u;
/*! \brief 8 bit signed char \ingroup group_rppdefs */
typedef signed char         Rpp8s;
/*! \brief 16 bit unsigned short \ingroup group_rppdefs */
typedef unsigned short      Rpp16u;
/*! \brief 16 bit signed short \ingroup group_rppdefs */
typedef short               Rpp16s;
/*! \brief 32 bit unsigned int \ingroup group_rppdefs */
typedef unsigned int        Rpp32u;
/*! \brief 32 bit signed int \ingroup group_rppdefs */
typedef int                 Rpp32s;
/*! \brief 64 bit unsigned long long \ingroup group_rppdefs */
typedef unsigned long long  Rpp64u;
/*! \brief 64 bit long long \ingroup group_rppdefs */
typedef long long           Rpp64s;
/*! \brief 32 bit float \ingroup group_rppdefs */
typedef float               Rpp32f;
/*! \brief 64 bit double \ingroup group_rppdefs */
typedef double              Rpp64f;
/*! \brief void pointer \ingroup group_rppdefs */
typedef void*               RppPtr_t;
/*! \brief size_t \ingroup group_rppdefs */
typedef size_t              RppSize_t;

typedef unsigned char       Npp8u;     /**<  8-bit unsigned chars */
typedef signed char         Npp8s;     /**<  8-bit signed chars */
typedef unsigned short      Npp16u;    /**<  16-bit unsigned integers */
typedef short               Npp16s;    /**<  16-bit signed integers */
typedef unsigned int        Npp32u;    /**<  32-bit unsigned integers */
typedef int                 Npp32s;    /**<  32-bit signed integers */
typedef unsigned long long  Npp64u;    /**<  64-bit unsigned integers */
typedef long long           Npp64s;    /**<  64-bit signed integers */
typedef float               Npp32f;    /**<  32-bit (IEEE) floating-point numbers */
typedef double              Npp64f;    /**<  64-bit floating-point numbers */

typedef struct
{
    const void * pSrc;  /* device memory pointer */
    int nSrcStep;
    void * pDst;        /* device memory pointer */
    int nDstStep;
    Npp32f * pTwist; /* device memory pointer to the color twist matrix with floating-point coefficient values to be used for this image */
} NppiColorTwistBatchCXR;

typedef 
struct
{
   short fp16;
}
Npp16f;

typedef struct 
{
    int x;      /**<  x-coordinate. */
    int y;      /**<  y-coordinate. */
} NppiPoint;

typedef enum
{
    NPP_MASK_SIZE_1_X_3,
    NPP_MASK_SIZE_1_X_5,
    NPP_MASK_SIZE_3_X_1 = 100, // leaving space for more 1 X N type enum values 
    NPP_MASK_SIZE_5_X_1,
    NPP_MASK_SIZE_3_X_3 = 200, // leaving space for more N X 1 type enum values
    NPP_MASK_SIZE_5_X_5,
    NPP_MASK_SIZE_7_X_7 = 400,
    NPP_MASK_SIZE_9_X_9 = 500,
    NPP_MASK_SIZE_11_X_11 = 600,
    NPP_MASK_SIZE_13_X_13 = 700,
    NPP_MASK_SIZE_15_X_15 = 800
} NppiMaskSize;

typedef enum {
    nppiNormInf = 0, /**<  maximum */ 
    nppiNormL1 = 1,  /**<  sum */
    nppiNormL2 = 2   /**<  square root of sum of squares */
} NppiNorm;

typedef enum {
    rppiNormInf = 0, /**<  maximum */ 
    rppiNormL1 = 1,  /**<  sum */
    rppiNormL2 = 2   /**<  square root of sum of squares */
} RppiNorm;

typedef enum 
{
    NPP_CMP_LESS,
    NPP_CMP_LESS_EQ,
    NPP_CMP_EQ,
    NPP_CMP_GREATER_EQ,
    NPP_CMP_GREATER
} NppCmpOp;

typedef enum 
{
    RPP_CMP_LESS,
    RPP_CMP_LESS_EQ,
    RPP_CMP_EQ,
    RPP_CMP_GREATER_EQ,
    RPP_CMP_GREATER
} RppCmpOp;

typedef enum
{
    NPP_FILTER_SOBEL,
    NPP_FILTER_SCHARR,
} NppiDifferentialKernel;

typedef struct
{
    int x;          /**<  x-coordinate of upper left corner (lowest memory address). */
    int y;          /**<  y-coordinate of upper left corner (lowest memory address). */
    int width;      /**<  Rectangle width. */
    int height;     /**<  Rectangle height. */
} NppiRect;

typedef struct
{
    int width;  /**<  Rectangle width. */
    int height; /**<  Rectangle height. */
} NppiSize;

typedef struct
{
    void *     pData;  /**< device memory pointer to the image */
    int        nStep;  /**< step size */
    NppiSize  oSize;   /**< width and height of the image */
} NppiImageDescriptor;

typedef struct
{
    void *     pData;        /**< per image device memory pointer to the corresponding buffer */
    int        nBufferSize;  /**< allocated buffer size */
} NppiBufferDescriptor;

typedef struct
{
    void *     pData;        /**< per image device memory pointer to the corresponding buffer */
    int        nBufferSize;  /**< allocated buffer size */
} RppBufferDescriptor;

typedef enum 
{
    NPPI_INTER_UNDEFINED         = 0,
    NPPI_INTER_NN                = 1,        /**<  Nearest neighbor filtering. */
    NPPI_INTER_LINEAR            = 2,        /**<  Linear interpolation. */
    NPPI_INTER_CUBIC             = 4,        /**<  Cubic interpolation. */
    NPPI_INTER_CUBIC2P_BSPLINE,              /**<  Two-parameter cubic filter (B=1, C=0) */
    NPPI_INTER_CUBIC2P_CATMULLROM,           /**<  Two-parameter cubic filter (B=0, C=1/2) */
    NPPI_INTER_CUBIC2P_B05C03,               /**<  Two-parameter cubic filter (B=1/2, C=3/10) */
    NPPI_INTER_SUPER             = 8,        /**<  Super sampling. */
    NPPI_INTER_LANCZOS           = 16,       /**<  Lanczos filtering. */
    NPPI_INTER_LANCZOS3_ADVANCED = 17,       /**<  Generic Lanczos filtering with order 3. */
    NPPI_SMOOTH_EDGE             = (int)0x8000000 /**<  Smooth edge filtering. */
} NppiInterpolationMode; 

typedef enum 
{
    RPPI_INTER_UNDEFINED         = 0,
    RPPI_INTER_NN                = 1,        /**<  Nearest neighbor filtering. */
    RPPI_INTER_LINEAR            = 2,        /**<  Linear interpolation. */
    RPPI_INTER_CUBIC             = 4,        /**<  Cubic interpolation. */
    RPPI_INTER_CUBIC2P_BSPLINE,              /**<  Two-parameter cubic filter (B=1, C=0) */
    RPPI_INTER_CUBIC2P_CATMULLROM,           /**<  Two-parameter cubic filter (B=0, C=1/2) */
    RPPI_INTER_CUBIC2P_B05C03,               /**<  Two-parameter cubic filter (B=1/2, C=3/10) */
    RPPI_INTER_SUPER             = 8,        /**<  Super sampling. */
    RPPI_INTER_LANCZOS           = 16,       /**<  Lanczos filtering. */
    RPPI_INTER_LANCZOS3_ADVANCED = 17,       /**<  Generic Lanczos filtering with order 3. */
    RPPI_SMOOTH_EDGE             = (int)0x8000000 /**<  Smooth edge filtering. */
} RppiInterpolationMode; 

typedef enum 
{
    NPPI_BAYER_BGGR         = 0,             /**<  Default registration position. */
    NPPI_BAYER_RGGB         = 1,
    NPPI_BAYER_GBRG         = 2,
    NPPI_BAYER_GRBG         = 3
} NppiBayerGridPosition; 

typedef enum 
{
    RPPI_BAYER_BGGR         = 0,             /**<  Default registration position. */
    RPPI_BAYER_RGGB         = 1,
    RPPI_BAYER_GBRG         = 2,
    RPPI_BAYER_GRBG         = 3
} RppiBayerGridPosition;

typedef enum 
{
    NPP_HORIZONTAL_AXIS,
    NPP_VERTICAL_AXIS,
    NPP_BOTH_AXIS
} NppiAxis;

/*! \brief RPP RppStatus type enums
 * \ingroup group_rppdefs
 */
typedef enum
{
    /*! \brief No error. \ingroup group_rppdefs */
    RPP_SUCCESS                         = 0,
    /*! \brief Unspecified error. \ingroup group_rppdefs */
    RPP_ERROR                           = -1,
    /*! \brief One or more arguments invalid. (Needs to adhere to function specification.) \ingroup group_rppdefs */
    RPP_ERROR_INVALID_ARGUMENTS         = -2,
    /*! \brief Low tensor offsetInBytes provided for src/dst tensor. \ingroup group_rppdefs */
    RPP_ERROR_LOW_OFFSET                = -3,
    /*! \brief Arguments provided will result in zero division error. \ingroup group_rppdefs */
    RPP_ERROR_ZERO_DIVISION             = -4,
    /*! \brief Src tensor / src ROI dimension too high. (Needs to adhere to function specification.) \ingroup group_rppdefs */
    RPP_ERROR_HIGH_SRC_DIMENSION        = -5,
    /*! \brief Function variant requested is not implemented / unsupported. \ingroup group_rppdefs */
    RPP_ERROR_NOT_IMPLEMENTED           = -6,
    /*! \brief Invalid src tensor number of channels. (Needs to adhere to function specification.) \ingroup group_rppdefs */
    RPP_ERROR_INVALID_SRC_CHANNELS      = -7,
    /*! \brief Invalid dst tensor number of channels. (Needs to adhere to function specification.) \ingroup group_rppdefs */
    RPP_ERROR_INVALID_DST_CHANNELS      = -8,
    /*! \brief Invalid src tensor layout. (Needs to adhere to function specification.) \ingroup group_rppdefs */
    RPP_ERROR_INVALID_SRC_LAYOUT        = -9,
    /*! \brief Invalid dst tensor layout. (Needs to adhere to function specification.) \ingroup group_rppdefs */
    RPP_ERROR_INVALID_DST_LAYOUT        = -10,
    /*! \brief Invalid src tensor datatype. (Needs to adhere to function specification.) \ingroup group_rppdefs */
    RPP_ERROR_INVALID_SRC_DATATYPE      = -11,
    /*! \brief Invalid dst tensor datatype. (Needs to adhere to function specification.) \ingroup group_rppdefs */
    RPP_ERROR_INVALID_DST_DATATYPE      = -12,
    /*! \brief Invalid src/dst tensor datatype. (Needs to adhere to function specification.) \ingroup group_rppdefs */
    RPP_ERROR_INVALID_SRC_OR_DST_DATATYPE       = -13,
    /*! \brief Insufficient dst buffer length provided. (Needs to adhere to function specification.) \ingroup group_rppdefs */
    RPP_ERROR_INSUFFICIENT_DST_BUFFER_LENGTH    = -14,
    /*! \brief Invalid datatype \ingroup group_rppdefs */
    RPP_ERROR_INVALID_PARAMETER_DATATYPE        = -15,
    /*! \brief Not enough memory to write outputs, as per dim-lengths and strides set in descriptor \ingroup group_rppdefs */
    RPP_ERROR_NOT_ENOUGH_MEMORY         = -16,
    /*! \brief Out of bound source ROI \ingroup group_rppdefs */
    RPP_ERROR_OUT_OF_BOUND_SRC_ROI      = -17,
    /*! \brief src and dst layout mismatch \ingroup group_rppdefs */
    RPP_ERROR_LAYOUT_MISMATCH           = -18,
    /*! \brief Number of channels is invalid. (Needs to adhere to function specification.) \ingroup group_rppdefs */
    RPP_ERROR_INVALID_CHANNELS          = -19,
    /*! \brief Invalid output tile length (Needs to adhere to function specification.) \ingroup group_rppdefs */
    RPP_ERROR_INVALID_OUTPUT_TILE_LENGTH    = -20,
    /*! \brief Shared memory size needed is beyond the bounds (Needs to adhere to function specification.) \ingroup group_rppdefs */
    RPP_ERROR_OUT_OF_BOUND_SHARED_MEMORY_SIZE    = -21,
    /*! \brief Scratch memory size needed is beyond the bounds (Needs to adhere to function specification.) \ingroup group_rppdefs */
    RPP_ERROR_OUT_OF_BOUND_SCRATCH_MEMORY_SIZE    = -22,
    /*! \brief Number of src dims is invalid. (Needs to adhere to function specification.) \ingroup group_rppdefs */
    RPP_ERROR_INVALID_SRC_DIMS          = -23,
    /*! \brief Number of dst dims is invalid. (Needs to adhere to function specification.) \ingroup group_rppdefs */
    RPP_ERROR_INVALID_DST_DIMS          = -24
} RppStatus;

typedef enum 
{
    /* negative return-codes indicate errors */
    NPP_NOT_SUPPORTED_MODE_ERROR            = -9999,  
    
    NPP_INVALID_HOST_POINTER_ERROR          = -1032,
    NPP_INVALID_DEVICE_POINTER_ERROR        = -1031,
    NPP_LUT_PALETTE_BITSIZE_ERROR           = -1030,
    NPP_ZC_MODE_NOT_SUPPORTED_ERROR         = -1028,      /**<  ZeroCrossing mode not supported  */
    NPP_NOT_SUFFICIENT_COMPUTE_CAPABILITY   = -1027,
    NPP_TEXTURE_BIND_ERROR                  = -1024,
    NPP_WRONG_INTERSECTION_ROI_ERROR        = -1020,
    NPP_HAAR_CLASSIFIER_PIXEL_MATCH_ERROR   = -1006,
    NPP_MEMFREE_ERROR                       = -1005,
    NPP_MEMSET_ERROR                        = -1004,
    NPP_MEMCPY_ERROR                        = -1003,
    NPP_ALIGNMENT_ERROR                     = -1002,
    NPP_CUDA_KERNEL_EXECUTION_ERROR         = -1000,

    NPP_ROUND_MODE_NOT_SUPPORTED_ERROR      = -213,     /**< Unsupported round mode*/
    
    NPP_QUALITY_INDEX_ERROR                 = -210,     /**< Image pixels are constant for quality index */

    NPP_RESIZE_NO_OPERATION_ERROR           = -201,     /**< One of the output image dimensions is less than 1 pixel */

    NPP_OVERFLOW_ERROR                      = -109,     /**< Number overflows the upper or lower limit of the data type */
    NPP_NOT_EVEN_STEP_ERROR                 = -108,     /**< Step value is not pixel multiple */
    NPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR    = -107,     /**< Number of levels for histogram is less than 2 */
    NPP_LUT_NUMBER_OF_LEVELS_ERROR          = -106,     /**< Number of levels for LUT is less than 2 */

    NPP_CORRUPTED_DATA_ERROR                = -61,      /**< Processed data is corrupted */
    NPP_CHANNEL_ORDER_ERROR                 = -60,      /**< Wrong order of the destination channels */
    NPP_ZERO_MASK_VALUE_ERROR               = -59,      /**< All values of the mask are zero */
    NPP_QUADRANGLE_ERROR                    = -58,      /**< The quadrangle is nonconvex or degenerates into triangle, line or point */
    NPP_RECTANGLE_ERROR                     = -57,      /**< Size of the rectangle region is less than or equal to 1 */
    NPP_COEFFICIENT_ERROR                   = -56,      /**< Unallowable values of the transformation coefficients   */

    NPP_NUMBER_OF_CHANNELS_ERROR            = -53,      /**< Bad or unsupported number of channels */
    NPP_COI_ERROR                           = -52,      /**< Channel of interest is not 1, 2, or 3 */
    NPP_DIVISOR_ERROR                       = -51,      /**< Divisor is equal to zero */

    NPP_CHANNEL_ERROR                       = -47,      /**< Illegal channel index */
    NPP_STRIDE_ERROR                        = -37,      /**< Stride is less than the row length */
    
    NPP_ANCHOR_ERROR                        = -34,      /**< Anchor point is outside mask */
    NPP_MASK_SIZE_ERROR                     = -33,      /**< Lower bound is larger than upper bound */

    NPP_RESIZE_FACTOR_ERROR                 = -23,
    NPP_INTERPOLATION_ERROR                 = -22,
    NPP_MIRROR_FLIP_ERROR                   = -21,
    NPP_MOMENT_00_ZERO_ERROR                = -20,
    NPP_THRESHOLD_NEGATIVE_LEVEL_ERROR      = -19,
    NPP_THRESHOLD_ERROR                     = -18,
    NPP_CONTEXT_MATCH_ERROR                 = -17,
    NPP_FFT_FLAG_ERROR                      = -16,
    NPP_FFT_ORDER_ERROR                     = -15,
    NPP_STEP_ERROR                          = -14,       /**<  Step is less or equal zero */
    NPP_SCALE_RANGE_ERROR                   = -13,
    NPP_DATA_TYPE_ERROR                     = -12,
    NPP_OUT_OFF_RANGE_ERROR                 = -11,
    NPP_DIVIDE_BY_ZERO_ERROR                = -10,
    NPP_MEMORY_ALLOCATION_ERR               = -9,
    NPP_NULL_POINTER_ERROR                  = -8,
    NPP_RANGE_ERROR                         = -7,
    NPP_SIZE_ERROR                          = -6,
    NPP_BAD_ARGUMENT_ERROR                  = -5,
    NPP_NO_MEMORY_ERROR                     = -4,
    NPP_NOT_IMPLEMENTED_ERROR               = -3,
    NPP_ERROR                               = -2,
    NPP_ERROR_RESERVED                      = -1,
    
    /* success */
    NPP_NO_ERROR                            = 0,        /**<  Error free operation */
    NPP_SUCCESS = NPP_NO_ERROR,                         /**<  Successful operation (same as NPP_NO_ERROR) */

    /* positive return-codes indicate warnings */
    NPP_NO_OPERATION_WARNING                = 1,        /**<  Indicates that no operation was performed */
    NPP_DIVIDE_BY_ZERO_WARNING              = 6,        /**<  Divisor is zero however does not terminate the execution */
    NPP_AFFINE_QUAD_INCORRECT_WARNING       = 28,       /**<  Indicates that the quadrangle passed to one of affine warping functions doesn't have necessary properties. First 3 vertices are used, the fourth vertex discarded. */
    NPP_WRONG_INTERSECTION_ROI_WARNING      = 29,       /**<  The given ROI has no interestion with either the source or destination ROI. Thus no operation was performed. */
    NPP_WRONG_INTERSECTION_QUAD_WARNING     = 30,       /**<  The given quadrangle has no intersection with either the source or destination ROI. Thus no operation was performed. */
    NPP_DOUBLE_SIZE_WARNING                 = 35,       /**<  Image size isn't multiple of two. Indicates that in case of 422/411/420 sampling the ROI width/height was modified for proper processing. */
    
    NPP_MISALIGNED_DST_ROI_WARNING          = 10000,    /**<  Speed reduction due to uncoalesced memory accesses warning. */
   
} NppStatus;

typedef enum  
{
    NPP_BORDER_UNDEFINED        = 0,
    NPP_BORDER_NONE             = NPP_BORDER_UNDEFINED, 
    NPP_BORDER_CONSTANT         = 1,
    NPP_BORDER_REPLICATE        = 2,
    NPP_BORDER_WRAP             = 3,
    NPP_BORDER_MIRROR           = 4
} NppiBorderType;

typedef enum  
{
    RPP_BORDER_UNDEFINED        = 0,
    RPP_BORDER_NONE             = RPP_BORDER_UNDEFINED, 
    RPP_BORDER_CONSTANT         = 1,
    RPP_BORDER_REPLICATE        = 2,
    RPP_BORDER_WRAP             = 3,
    RPP_BORDER_MIRROR           = 4
} RppiBorderType;

enum cudaError
{
    /**
     * The API call returned with no errors. In the case of query calls, this
     * also means that the operation being queried is complete (see
     * ::cudaEventQuery() and ::cudaStreamQuery()).
     */
    cudaSuccess                           =      0,
  
    /**
     * This indicates that one or more of the parameters passed to the API call
     * is not within an acceptable range of values.
     */
    cudaErrorInvalidValue                 =     1,
  
    /**
     * The API call failed because it was unable to allocate enough memory to
     * perform the requested operation.
     */
    cudaErrorMemoryAllocation             =      2,
  
    /**
     * The API call failed because the CUDA driver and runtime could not be
     * initialized.
     */
    cudaErrorInitializationError          =      3,
  
    /**
     * This indicates that a CUDA Runtime API call cannot be executed because
     * it is being called during process shut down, at a point in time after
     * CUDA driver has been unloaded.
     */
    cudaErrorCudartUnloading              =     4,

    /**
     * This indicates profiler is not initialized for this run. This can
     * happen when the application is running with external profiling tools
     * like visual profiler.
     */
    cudaErrorProfilerDisabled             =     5,

    /**
     * \deprecated
     * This error return is deprecated as of CUDA 5.0. It is no longer an error
     * to attempt to enable/disable the profiling via ::cudaProfilerStart or
     * ::cudaProfilerStop without initialization.
     */
    cudaErrorProfilerNotInitialized       =     6,

    /**
     * \deprecated
     * This error return is deprecated as of CUDA 5.0. It is no longer an error
     * to call cudaProfilerStart() when profiling is already enabled.
     */
    cudaErrorProfilerAlreadyStarted       =     7,

    /**
     * \deprecated
     * This error return is deprecated as of CUDA 5.0. It is no longer an error
     * to call cudaProfilerStop() when profiling is already disabled.
     */
     cudaErrorProfilerAlreadyStopped       =    8,
  
    /**
     * This indicates that a kernel launch is requesting resources that can
     * never be satisfied by the current device. Requesting more shared memory
     * per block than the device supports will trigger this error, as will
     * requesting too many threads or blocks. See ::cudaDeviceProp for more
     * device limitations.
     */
    cudaErrorInvalidConfiguration         =      9,
  
    /**
     * This indicates that one or more of the pitch-related parameters passed
     * to the API call is not within the acceptable range for pitch.
     */
    cudaErrorInvalidPitchValue            =     12,
  
    /**
     * This indicates that the symbol name/identifier passed to the API call
     * is not a valid name or identifier.
     */
    cudaErrorInvalidSymbol                =     13,
  
    /**
     * This indicates that at least one host pointer passed to the API call is
     * not a valid host pointer.
     * \deprecated
     * This error return is deprecated as of CUDA 10.1.
     */
    cudaErrorInvalidHostPointer           =     16,
  
    /**
     * This indicates that at least one device pointer passed to the API call is
     * not a valid device pointer.
     * \deprecated
     * This error return is deprecated as of CUDA 10.1.
     */
    cudaErrorInvalidDevicePointer         =     17,
  
    /**
     * This indicates that the texture passed to the API call is not a valid
     * texture.
     */
    cudaErrorInvalidTexture               =     18,
  
    /**
     * This indicates that the texture binding is not valid. This occurs if you
     * call ::cudaGetTextureAlignmentOffset() with an unbound texture.
     */
    cudaErrorInvalidTextureBinding        =     19,
  
    /**
     * This indicates that the channel descriptor passed to the API call is not
     * valid. This occurs if the format is not one of the formats specified by
     * ::cudaChannelFormatKind, or if one of the dimensions is invalid.
     */
    cudaErrorInvalidChannelDescriptor     =     20,
  
    /**
     * This indicates that the direction of the memcpy passed to the API call is
     * not one of the types specified by ::cudaMemcpyKind.
     */
    cudaErrorInvalidMemcpyDirection       =     21,
  
    /**
     * This indicated that the user has taken the address of a constant variable,
     * which was forbidden up until the CUDA 3.1 release.
     * \deprecated
     * This error return is deprecated as of CUDA 3.1. Variables in constant
     * memory may now have their address taken by the runtime via
     * ::cudaGetSymbolAddress().
     */
    cudaErrorAddressOfConstant            =     22,
  
    /**
     * This indicated that a texture fetch was not able to be performed.
     * This was previously used for device emulation of texture operations.
     * \deprecated
     * This error return is deprecated as of CUDA 3.1. Device emulation mode was
     * removed with the CUDA 3.1 release.
     */
    cudaErrorTextureFetchFailed           =     23,
  
    /**
     * This indicated that a texture was not bound for access.
     * This was previously used for device emulation of texture operations.
     * \deprecated
     * This error return is deprecated as of CUDA 3.1. Device emulation mode was
     * removed with the CUDA 3.1 release.
     */
    cudaErrorTextureNotBound              =     24,
  
    /**
     * This indicated that a synchronization operation had failed.
     * This was previously used for some device emulation functions.
     * \deprecated
     * This error return is deprecated as of CUDA 3.1. Device emulation mode was
     * removed with the CUDA 3.1 release.
     */
    cudaErrorSynchronizationError         =     25,
  
    /**
     * This indicates that a non-float texture was being accessed with linear
     * filtering. This is not supported by CUDA.
     */
    cudaErrorInvalidFilterSetting         =     26,
  
    /**
     * This indicates that an attempt was made to read a non-float texture as a
     * normalized float. This is not supported by CUDA.
     */
    cudaErrorInvalidNormSetting           =     27,
  
    /**
     * Mixing of device and device emulation code was not allowed.
     * \deprecated
     * This error return is deprecated as of CUDA 3.1. Device emulation mode was
     * removed with the CUDA 3.1 release.
     */
    cudaErrorMixedDeviceExecution         =     28,

    /**
     * This indicates that the API call is not yet implemented. Production
     * releases of CUDA will never return this error.
     * \deprecated
     * This error return is deprecated as of CUDA 4.1.
     */
    cudaErrorNotYetImplemented            =     31,
  
    /**
     * This indicated that an emulated device pointer exceeded the 32-bit address
     * range.
     * \deprecated
     * This error return is deprecated as of CUDA 3.1. Device emulation mode was
     * removed with the CUDA 3.1 release.
     */
    cudaErrorMemoryValueTooLarge          =     32,
  
    /**
     * This indicates that the CUDA driver that the application has loaded is a
     * stub library. Applications that run with the stub rather than a real
     * driver loaded will result in CUDA API returning this error.
     */
    cudaErrorStubLibrary                  =     34,

    /**
     * This indicates that the installed NVIDIA CUDA driver is older than the
     * CUDA runtime library. This is not a supported configuration. Users should
     * install an updated NVIDIA display driver to allow the application to run.
     */
    cudaErrorInsufficientDriver           =     35,

    /**
     * This indicates that the API call requires a newer CUDA driver than the one
     * currently installed. Users should install an updated NVIDIA CUDA driver
     * to allow the API call to succeed.
     */
    cudaErrorCallRequiresNewerDriver      =     36,
  
    /**
     * This indicates that the surface passed to the API call is not a valid
     * surface.
     */
    cudaErrorInvalidSurface               =     37,
  
    /**
     * This indicates that multiple global or constant variables (across separate
     * CUDA source files in the application) share the same string name.
     */
    cudaErrorDuplicateVariableName        =     43,
  
    /**
     * This indicates that multiple textures (across separate CUDA source
     * files in the application) share the same string name.
     */
    cudaErrorDuplicateTextureName         =     44,
  
    /**
     * This indicates that multiple surfaces (across separate CUDA source
     * files in the application) share the same string name.
     */
    cudaErrorDuplicateSurfaceName         =     45,
  
    /**
     * This indicates that all CUDA devices are busy or unavailable at the current
     * time. Devices are often busy/unavailable due to use of
     * ::cudaComputeModeProhibited, ::cudaComputeModeExclusiveProcess, or when long
     * running CUDA kernels have filled up the GPU and are blocking new work
     * from starting. They can also be unavailable due to memory constraints
     * on a device that already has active CUDA work being performed.
     */
    cudaErrorDevicesUnavailable           =     46,
  
    /**
     * This indicates that the current context is not compatible with this
     * the CUDA Runtime. This can only occur if you are using CUDA
     * Runtime/Driver interoperability and have created an existing Driver
     * context using the driver API. The Driver context may be incompatible
     * either because the Driver context was created using an older version 
     * of the API, because the Runtime API call expects a primary driver 
     * context and the Driver context is not primary, or because the Driver 
     * context has been destroyed. Please see \ref CUDART_DRIVER "Interactions 
     * with the CUDA Driver API" for more information.
     */
    cudaErrorIncompatibleDriverContext    =     49,
    
    /**
     * The device function being invoked (usually via ::cudaLaunchKernel()) was not
     * previously configured via the ::cudaConfigureCall() function.
     */
    cudaErrorMissingConfiguration         =      52,
  
    /**
     * This indicated that a previous kernel launch failed. This was previously
     * used for device emulation of kernel launches.
     * \deprecated
     * This error return is deprecated as of CUDA 3.1. Device emulation mode was
     * removed with the CUDA 3.1 release.
     */
    cudaErrorPriorLaunchFailure           =      53,

    /**
     * This error indicates that a device runtime grid launch did not occur 
     * because the depth of the child grid would exceed the maximum supported
     * number of nested grid launches. 
     */
    cudaErrorLaunchMaxDepthExceeded       =     65,

    /**
     * This error indicates that a grid launch did not occur because the kernel 
     * uses file-scoped textures which are unsupported by the device runtime. 
     * Kernels launched via the device runtime only support textures created with 
     * the Texture Object API's.
     */
    cudaErrorLaunchFileScopedTex          =     66,

    /**
     * This error indicates that a grid launch did not occur because the kernel 
     * uses file-scoped surfaces which are unsupported by the device runtime.
     * Kernels launched via the device runtime only support surfaces created with
     * the Surface Object API's.
     */
    cudaErrorLaunchFileScopedSurf         =     67,

    /**
     * This error indicates that a call to ::cudaDeviceSynchronize made from
     * the device runtime failed because the call was made at grid depth greater
     * than than either the default (2 levels of grids) or user specified device 
     * limit ::cudaLimitDevRuntimeSyncDepth. To be able to synchronize on 
     * launched grids at a greater depth successfully, the maximum nested 
     * depth at which ::cudaDeviceSynchronize will be called must be specified 
     * with the ::cudaLimitDevRuntimeSyncDepth limit to the ::cudaDeviceSetLimit
     * api before the host-side launch of a kernel using the device runtime. 
     * Keep in mind that additional levels of sync depth require the runtime 
     * to reserve large amounts of device memory that cannot be used for 
     * user allocations.
     */
    cudaErrorSyncDepthExceeded            =     68,

    /**
     * This error indicates that a device runtime grid launch failed because
     * the launch would exceed the limit ::cudaLimitDevRuntimePendingLaunchCount.
     * For this launch to proceed successfully, ::cudaDeviceSetLimit must be
     * called to set the ::cudaLimitDevRuntimePendingLaunchCount to be higher 
     * than the upper bound of outstanding launches that can be issued to the
     * device runtime. Keep in mind that raising the limit of pending device
     * runtime launches will require the runtime to reserve device memory that
     * cannot be used for user allocations.
     */
    cudaErrorLaunchPendingCountExceeded   =     69,
  
    /**
     * The requested device function does not exist or is not compiled for the
     * proper device architecture.
     */
    cudaErrorInvalidDeviceFunction        =      98,
  
    /**
     * This indicates that no CUDA-capable devices were detected by the installed
     * CUDA driver.
     */
    cudaErrorNoDevice                     =     100,
  
    /**
     * This indicates that the device ordinal supplied by the user does not
     * correspond to a valid CUDA device or that the action requested is
     * invalid for the specified device.
     */
    cudaErrorInvalidDevice                =     101,

    /**
     * This indicates that the device doesn't have a valid Grid License.
     */
    cudaErrorDeviceNotLicensed            =     102,

   /**
    * By default, the CUDA runtime may perform a minimal set of self-tests,
    * as well as CUDA driver tests, to establish the validity of both.
    * Introduced in CUDA 11.2, this error return indicates that at least one
    * of these tests has failed and the validity of either the runtime
    * or the driver could not be established.
    */
   cudaErrorSoftwareValidityNotEstablished  =     103,

    /**
     * This indicates an internal startup failure in the CUDA runtime.
     */
    cudaErrorStartupFailure               =    127,
  
    /**
     * This indicates that the device kernel image is invalid.
     */
    cudaErrorInvalidKernelImage           =     200,

    /**
     * This most frequently indicates that there is no context bound to the
     * current thread. This can also be returned if the context passed to an
     * API call is not a valid handle (such as a context that has had
     * ::cuCtxDestroy() invoked on it). This can also be returned if a user
     * mixes different API versions (i.e. 3010 context with 3020 API calls).
     * See ::cuCtxGetApiVersion() for more details.
     */
    cudaErrorDeviceUninitialized          =     201,

    /**
     * This indicates that the buffer object could not be mapped.
     */
    cudaErrorMapBufferObjectFailed        =     205,
  
    /**
     * This indicates that the buffer object could not be unmapped.
     */
    cudaErrorUnmapBufferObjectFailed      =     206,

    /**
     * This indicates that the specified array is currently mapped and thus
     * cannot be destroyed.
     */
    cudaErrorArrayIsMapped                =     207,

    /**
     * This indicates that the resource is already mapped.
     */
    cudaErrorAlreadyMapped                =     208,
  
    /**
     * This indicates that there is no kernel image available that is suitable
     * for the device. This can occur when a user specifies code generation
     * options for a particular CUDA source file that do not include the
     * corresponding device configuration.
     */
    cudaErrorNoKernelImageForDevice       =     209,

    /**
     * This indicates that a resource has already been acquired.
     */
    cudaErrorAlreadyAcquired              =     210,

    /**
     * This indicates that a resource is not mapped.
     */
    cudaErrorNotMapped                    =     211,

    /**
     * This indicates that a mapped resource is not available for access as an
     * array.
     */
    cudaErrorNotMappedAsArray             =     212,

    /**
     * This indicates that a mapped resource is not available for access as a
     * pointer.
     */
    cudaErrorNotMappedAsPointer           =     213,
  
    /**
     * This indicates that an uncorrectable ECC error was detected during
     * execution.
     */
    cudaErrorECCUncorrectable             =     214,
  
    /**
     * This indicates that the ::cudaLimit passed to the API call is not
     * supported by the active device.
     */
    cudaErrorUnsupportedLimit             =     215,
    
    /**
     * This indicates that a call tried to access an exclusive-thread device that 
     * is already in use by a different thread.
     */
    cudaErrorDeviceAlreadyInUse           =     216,

    /**
     * This error indicates that P2P access is not supported across the given
     * devices.
     */
    cudaErrorPeerAccessUnsupported        =     217,

    /**
     * A PTX compilation failed. The runtime may fall back to compiling PTX if
     * an application does not contain a suitable binary for the current device.
     */
    cudaErrorInvalidPtx                   =     218,

    /**
     * This indicates an error with the OpenGL or DirectX context.
     */
    cudaErrorInvalidGraphicsContext       =     219,

    /**
     * This indicates that an uncorrectable NVLink error was detected during the
     * execution.
     */
    cudaErrorNvlinkUncorrectable          =     220,

    /**
     * This indicates that the PTX JIT compiler library was not found. The JIT Compiler
     * library is used for PTX compilation. The runtime may fall back to compiling PTX
     * if an application does not contain a suitable binary for the current device.
     */
    cudaErrorJitCompilerNotFound          =     221,

    /**
     * This indicates that the provided PTX was compiled with an unsupported toolchain.
     * The most common reason for this, is the PTX was generated by a compiler newer
     * than what is supported by the CUDA driver and PTX JIT compiler.
     */
    cudaErrorUnsupportedPtxVersion        =     222,

    /**
     * This indicates that the JIT compilation was disabled. The JIT compilation compiles
     * PTX. The runtime may fall back to compiling PTX if an application does not contain
     * a suitable binary for the current device.
     */
    cudaErrorJitCompilationDisabled       =     223,

    /**
     * This indicates that the provided execution affinity is not supported by the device.
     */
    cudaErrorUnsupportedExecAffinity      =     224,

    /**
     * This indicates that the device kernel source is invalid.
     */
    cudaErrorInvalidSource                =     300,

    /**
     * This indicates that the file specified was not found.
     */
    cudaErrorFileNotFound                 =     301,
  
    /**
     * This indicates that a link to a shared object failed to resolve.
     */
    cudaErrorSharedObjectSymbolNotFound   =     302,
  
    /**
     * This indicates that initialization of a shared object failed.
     */
    cudaErrorSharedObjectInitFailed       =     303,

    /**
     * This error indicates that an OS call failed.
     */
    cudaErrorOperatingSystem              =     304,
  
    /**
     * This indicates that a resource handle passed to the API call was not
     * valid. Resource handles are opaque types like ::cudaStream_t and
     * ::cudaEvent_t.
     */
    cudaErrorInvalidResourceHandle        =     400,

    /**
     * This indicates that a resource required by the API call is not in a
     * valid state to perform the requested operation.
     */
    cudaErrorIllegalState                 =     401,

    /**
     * This indicates that a named symbol was not found. Examples of symbols
     * are global/constant variable names, driver function names, texture names,
     * and surface names.
     */
    cudaErrorSymbolNotFound               =     500,
  
    /**
     * This indicates that asynchronous operations issued previously have not
     * completed yet. This result is not actually an error, but must be indicated
     * differently than ::cudaSuccess (which indicates completion). Calls that
     * may return this value include ::cudaEventQuery() and ::cudaStreamQuery().
     */
    cudaErrorNotReady                     =     600,

    /**
     * The device encountered a load or store instruction on an invalid memory address.
     * This leaves the process in an inconsistent state and any further CUDA work
     * will return the same error. To continue using CUDA, the process must be terminated
     * and relaunched.
     */
    cudaErrorIllegalAddress               =     700,
  
    /**
     * This indicates that a launch did not occur because it did not have
     * appropriate resources. Although this error is similar to
     * ::cudaErrorInvalidConfiguration, this error usually indicates that the
     * user has attempted to pass too many arguments to the device kernel, or the
     * kernel launch specifies too many threads for the kernel's register count.
     */
    cudaErrorLaunchOutOfResources         =      701,
  
    /**
     * This indicates that the device kernel took too long to execute. This can
     * only occur if timeouts are enabled - see the device property
     * \ref ::cudaDeviceProp::kernelExecTimeoutEnabled "kernelExecTimeoutEnabled"
     * for more information.
     * This leaves the process in an inconsistent state and any further CUDA work
     * will return the same error. To continue using CUDA, the process must be terminated
     * and relaunched.
     */
    cudaErrorLaunchTimeout                =      702,

    /**
     * This error indicates a kernel launch that uses an incompatible texturing
     * mode.
     */
    cudaErrorLaunchIncompatibleTexturing  =     703,
      
    /**
     * This error indicates that a call to ::cudaDeviceEnablePeerAccess() is
     * trying to re-enable peer addressing on from a context which has already
     * had peer addressing enabled.
     */
    cudaErrorPeerAccessAlreadyEnabled     =     704,
    
    /**
     * This error indicates that ::cudaDeviceDisablePeerAccess() is trying to 
     * disable peer addressing which has not been enabled yet via 
     * ::cudaDeviceEnablePeerAccess().
     */
    cudaErrorPeerAccessNotEnabled         =     705,
  
    /**
     * This indicates that the user has called ::cudaSetValidDevices(),
     * ::cudaSetDeviceFlags(), ::cudaD3D9SetDirect3DDevice(),
     * ::cudaD3D10SetDirect3DDevice, ::cudaD3D11SetDirect3DDevice(), or
     * ::cudaVDPAUSetVDPAUDevice() after initializing the CUDA runtime by
     * calling non-device management operations (allocating memory and
     * launching kernels are examples of non-device management operations).
     * This error can also be returned if using runtime/driver
     * interoperability and there is an existing ::CUcontext active on the
     * host thread.
     */
    cudaErrorSetOnActiveProcess           =     708,

    /**
     * This error indicates that the context current to the calling thread
     * has been destroyed using ::cuCtxDestroy, or is a primary context which
     * has not yet been initialized.
     */
    cudaErrorContextIsDestroyed           =     709,

    /**
     * An assert triggered in device code during kernel execution. The device
     * cannot be used again. All existing allocations are invalid. To continue
     * using CUDA, the process must be terminated and relaunched.
     */
    cudaErrorAssert                        =    710,
  
    /**
     * This error indicates that the hardware resources required to enable
     * peer access have been exhausted for one or more of the devices 
     * passed to ::cudaEnablePeerAccess().
     */
    cudaErrorTooManyPeers                 =     711,
  
    /**
     * This error indicates that the memory range passed to ::cudaHostRegister()
     * has already been registered.
     */
    cudaErrorHostMemoryAlreadyRegistered  =     712,
        
    /**
     * This error indicates that the pointer passed to ::cudaHostUnregister()
     * does not correspond to any currently registered memory region.
     */
    cudaErrorHostMemoryNotRegistered      =     713,

    /**
     * Device encountered an error in the call stack during kernel execution,
     * possibly due to stack corruption or exceeding the stack size limit.
     * This leaves the process in an inconsistent state and any further CUDA work
     * will return the same error. To continue using CUDA, the process must be terminated
     * and relaunched.
     */
    cudaErrorHardwareStackError           =     714,

    /**
     * The device encountered an illegal instruction during kernel execution
     * This leaves the process in an inconsistent state and any further CUDA work
     * will return the same error. To continue using CUDA, the process must be terminated
     * and relaunched.
     */
    cudaErrorIllegalInstruction           =     715,

    /**
     * The device encountered a load or store instruction
     * on a memory address which is not aligned.
     * This leaves the process in an inconsistent state and any further CUDA work
     * will return the same error. To continue using CUDA, the process must be terminated
     * and relaunched.
     */
    cudaErrorMisalignedAddress            =     716,

    /**
     * While executing a kernel, the device encountered an instruction
     * which can only operate on memory locations in certain address spaces
     * (global, shared, or local), but was supplied a memory address not
     * belonging to an allowed address space.
     * This leaves the process in an inconsistent state and any further CUDA work
     * will return the same error. To continue using CUDA, the process must be terminated
     * and relaunched.
     */
    cudaErrorInvalidAddressSpace          =     717,

    /**
     * The device encountered an invalid program counter.
     * This leaves the process in an inconsistent state and any further CUDA work
     * will return the same error. To continue using CUDA, the process must be terminated
     * and relaunched.
     */
    cudaErrorInvalidPc                    =     718,
  
    /**
     * An exception occurred on the device while executing a kernel. Common
     * causes include dereferencing an invalid device pointer and accessing
     * out of bounds shared memory. Less common cases can be system specific - more
     * information about these cases can be found in the system specific user guide.
     * This leaves the process in an inconsistent state and any further CUDA work
     * will return the same error. To continue using CUDA, the process must be terminated
     * and relaunched.
     */
    cudaErrorLaunchFailure                =      719,

    /**
     * This error indicates that the number of blocks launched per grid for a kernel that was
     * launched via either ::cudaLaunchCooperativeKernel or ::cudaLaunchCooperativeKernelMultiDevice
     * exceeds the maximum number of blocks as allowed by ::cudaOccupancyMaxActiveBlocksPerMultiprocessor
     * or ::cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags times the number of multiprocessors
     * as specified by the device attribute ::cudaDevAttrMultiProcessorCount.
     */
    cudaErrorCooperativeLaunchTooLarge    =     720,
    
    /**
     * This error indicates the attempted operation is not permitted.
     */
    cudaErrorNotPermitted                 =     800,

    /**
     * This error indicates the attempted operation is not supported
     * on the current system or device.
     */
    cudaErrorNotSupported                 =     801,

    /**
     * This error indicates that the system is not yet ready to start any CUDA
     * work.  To continue using CUDA, verify the system configuration is in a
     * valid state and all required driver daemons are actively running.
     * More information about this error can be found in the system specific
     * user guide.
     */
    cudaErrorSystemNotReady               =     802,

    /**
     * This error indicates that there is a mismatch between the versions of
     * the display driver and the CUDA driver. Refer to the compatibility documentation
     * for supported versions.
     */
    cudaErrorSystemDriverMismatch         =     803,

    /**
     * This error indicates that the system was upgraded to run with forward compatibility
     * but the visible hardware detected by CUDA does not support this configuration.
     * Refer to the compatibility documentation for the supported hardware matrix or ensure
     * that only supported hardware is visible during initialization via the CUDA_VISIBLE_DEVICES
     * environment variable.
     */
    cudaErrorCompatNotSupportedOnDevice   =     804,

    /**
     * This error indicates that the MPS client failed to connect to the MPS control daemon or the MPS server.
     */
    cudaErrorMpsConnectionFailed          =     805,

    /**
     * This error indicates that the remote procedural call between the MPS server and the MPS client failed.
     */
    cudaErrorMpsRpcFailure                =     806,

    /**
     * This error indicates that the MPS server is not ready to accept new MPS client requests.
     * This error can be returned when the MPS server is in the process of recovering from a fatal failure.
     */
    cudaErrorMpsServerNotReady            =     807,

    /**
     * This error indicates that the hardware resources required to create MPS client have been exhausted.
     */
    cudaErrorMpsMaxClientsReached         =     808,

    /**
     * This error indicates the the hardware resources required to device connections have been exhausted.
     */
    cudaErrorMpsMaxConnectionsReached     =     809,

    /**
     * This error indicates that the MPS client has been terminated by the server. To continue using CUDA, the process must be terminated and relaunched.
     */
    cudaErrorMpsClientTerminated          =     810,

    /**
     * The operation is not permitted when the stream is capturing.
     */
    cudaErrorStreamCaptureUnsupported     =    900,

    /**
     * The current capture sequence on the stream has been invalidated due to
     * a previous error.
     */
    cudaErrorStreamCaptureInvalidated     =    901,

    /**
     * The operation would have resulted in a merge of two independent capture
     * sequences.
     */
    cudaErrorStreamCaptureMerge           =    902,

    /**
     * The capture was not initiated in this stream.
     */
    cudaErrorStreamCaptureUnmatched       =    903,

    /**
     * The capture sequence contains a fork that was not joined to the primary
     * stream.
     */
    cudaErrorStreamCaptureUnjoined        =    904,

    /**
     * A dependency would have been created which crosses the capture sequence
     * boundary. Only implicit in-stream ordering dependencies are allowed to
     * cross the boundary.
     */
    cudaErrorStreamCaptureIsolation       =    905,

    /**
     * The operation would have resulted in a disallowed implicit dependency on
     * a current capture sequence from cudaStreamLegacy.
     */
    cudaErrorStreamCaptureImplicit        =    906,

    /**
     * The operation is not permitted on an event which was last recorded in a
     * capturing stream.
     */
    cudaErrorCapturedEvent                =    907,
  
    /**
     * A stream capture sequence not initiated with the ::cudaStreamCaptureModeRelaxed
     * argument to ::cudaStreamBeginCapture was passed to ::cudaStreamEndCapture in a
     * different thread.
     */
    cudaErrorStreamCaptureWrongThread     =    908,

    /**
     * This indicates that the wait operation has timed out.
     */
    cudaErrorTimeout                      =    909,

    /**
     * This error indicates that the graph update was not performed because it included 
     * changes which violated constraints specific to instantiated graph update.
     */
    cudaErrorGraphExecUpdateFailure       =    910,

    /**
     * This indicates that an async error has occurred in a device outside of CUDA.
     * If CUDA was waiting for an external device's signal before consuming shared data,
     * the external device signaled an error indicating that the data is not valid for
     * consumption. This leaves the process in an inconsistent state and any further CUDA
     * work will return the same error. To continue using CUDA, the process must be
     * terminated and relaunched.
     */
    cudaErrorExternalDevice               =    911,

    /**
     * This indicates that a kernel launch error has occurred due to cluster
     * misconfiguration.
     */
    cudaErrorInvalidClusterSize           =    912,

    /**
     * This indicates that an unknown internal error has occurred.
     */
    cudaErrorUnknown                      =    999,

    /**
     * Any unhandled CUDA driver error is added to this value and returned via
     * the runtime. Production releases of CUDA should not return such errors.
     * \deprecated
     * This error return is deprecated as of CUDA 4.1.
     */
    cudaErrorApiFailureBase               =  10000
};

typedef struct   
{  
    int major;
    int minor;
    int build;
} NppLibraryVersion;

enum cudaMemcpyKind
{
    cudaMemcpyHostToHost          =   0,      /**< Host   -> Host */
    cudaMemcpyHostToDevice        =   1,      /**< Host   -> Device */
    cudaMemcpyDeviceToHost        =   2,      /**< Device -> Host */
    cudaMemcpyDeviceToDevice      =   3,      /**< Device -> Device */
    cudaMemcpyDefault             =   4       /**< Direction of the transfer is inferred from the pointer values. Requires unified virtual addressing */
};

typedef struct CUstream_st *cudaStream_t;

typedef struct
{
    cudaStream_t hStream;
    int nCudaDeviceId; /* From cudaGetDevice() */
    int nMultiProcessorCount; /* From cudaGetDeviceProperties() */
    int nMaxThreadsPerMultiProcessor; /* From cudaGetDeviceProperties() */
    int nMaxThreadsPerBlock; /* From cudaGetDeviceProperties() */
    size_t nSharedMemPerBlock; /* From cudaGetDeviceProperties */
    int nCudaDevAttrComputeCapabilityMajor; /* From cudaGetDeviceAttribute() */
    int nCudaDevAttrComputeCapabilityMinor; /* From cudaGetDeviceAttribute() */
    unsigned int nStreamFlags; /* From cudaStreamGetFlags() */
    int nReserved0;
} NppStreamContext;
/*! \brief RPP rppStatus_t type enums
 * \ingroup group_rppdefs
 */
typedef enum
{
    rppStatusSuccess        = 0,
    rppStatusBadParm        = -1,
    rppStatusUnknownError   = -2,
    rppStatusNotInitialized = -3,
    rppStatusInvalidValue   = -4,
    rppStatusAllocFailed    = -5,
    rppStatusInternalError  = -6,
    rppStatusNotImplemented = -7,
    rppStatusUnsupportedOp  = -8,
} rppStatus_t;

/*! \brief RPP Operations type enum
 * \ingroup group_rppdefs
 */
typedef enum
{
    RPP_SCALAR_OP_AND       = 1,
    RPP_SCALAR_OP_OR,
    RPP_SCALAR_OP_XOR,
    RPP_SCALAR_OP_NAND,
    RPP_SCALAR_OP_EQUAL,
    RPP_SCALAR_OP_NOTEQUAL,
    RPP_SCALAR_OP_LESS,
    RPP_SCALAR_OP_LESSEQ,
    RPP_SCALAR_OP_GREATER,
    RPP_SCALAR_OP_GREATEREQ,
    RPP_SCALAR_OP_ADD,
    RPP_SCALAR_OP_SUBTRACT,
    RPP_SCALAR_OP_MULTIPLY,
    RPP_SCALAR_OP_DIVIDE,
    RPP_SCALAR_OP_MODULUS,
    RPP_SCALAR_OP_MIN,
    RPP_SCALAR_OP_MAX,
} RppOp;

/*! \brief RPP BitDepth Conversion type enum
 * \ingroup group_rppdefs
 */
typedef enum
{
    U8_S8,
    S8_U8,
} RppConvertBitDepthMode;

/*! \brief RPP polar point
 * \ingroup group_rppdefs
 */
typedef struct
{
    Rpp32f rho;
    Rpp32f theta;
} RppPointPolar;

/*! \brief RPP layout params
 * \ingroup group_rppdefs
 */
typedef struct
{
    Rpp32u channelParam;
    Rpp32u bufferMultiplier;
} RppLayoutParams;

/*! \brief RPP 6 float vector
 * \ingroup group_rppdefs
 */
typedef struct
{
    Rpp32f data[6];
} Rpp32f6;

/*! \brief RPP 9 float vector
 * \ingroup group_rppdefs
 */
typedef struct
{
    Rpp32f data[9];
} Rpp32f9;

/*! \brief RPP 24 signed int vector
 * \ingroup group_rppdefs
 */
typedef struct
{
    Rpp32s data[24];
} Rpp32s24;

/*! \brief RPP 24 float vector
 * \ingroup group_rppdefs
 */
typedef struct
{
    Rpp32f data[24];
} Rpp32f24;

/******************** RPPI typedefs ********************/

/*! \brief RPPI Image color convert mode type enum
 * \ingroup group_rppdefs
 */
typedef enum
{
    RGB_HSV                 = 1,
    HSV_RGB
} RppiColorConvertMode;

/*! \brief RPPI Image fuzzy level type enum
 * \ingroup group_rppdefs
 */
typedef enum
{
    RPPI_LOW,
    RPPI_MEDIUM,
    RPPI_HIGH
} RppiFuzzyLevel;

/*! \brief RPPI Image channel format type enum
 * \ingroup group_rppdefs
 */
typedef enum
{
    RPPI_CHN_PLANAR,
    RPPI_CHN_PACKED
} RppiChnFormat;

/*! \brief RPP Image axis type enum
 * \ingroup group_rppdefs
 */
typedef enum
{
    RPPI_HORIZONTAL_AXIS,
    RPPI_VERTICAL_AXIS,
    RPPI_BOTH_AXIS
} RppiAxis;

/*! \brief RPPI Image blur type enum
 * \ingroup group_rppdefs
 */
typedef enum
{
    GAUSS3,
    GAUSS5,
    GAUSS3x1,
    GAUSS1x3,
    AVG3 = 10,
    AVG5
} RppiBlur;

/*! \brief RPPI Image pad type enum
 * \ingroup group_rppdefs
 */
typedef enum
{
    ZEROPAD,
    NOPAD
} RppiPad;

/*! \brief RPPI Image format type enum
 * \ingroup group_rppdefs
 */
typedef enum
{
    RGB,
    HSV
} RppiFormat;

/*! \brief RPPI Image size(Width/Height dimensions) type struct
 * \ingroup group_rppdefs
 */
typedef struct
{
    unsigned int width;
    unsigned int height;
} RppiSize;


/*! \brief RPPI Image 2D cartesian point type struct
 * \ingroup group_rppdefs
 */
typedef struct
{
    int x;
    int y;
} RppiPoint;

/*! \brief RPPI Image 3D point type struct
 * \ingroup group_rppdefs
 */
typedef struct
{
    int x;
    int y;
    int z;
} RppiPoint3D;

/*! \brief RPPI Image 2D Rectangle (XYWH format) type struct
 * \ingroup group_rppdefs
 */
typedef struct
{
    int x;
    int y;
    int width;
    int height;
} RppiRect;

/*! \brief RPPI Image 2D ROI (XYWH format) type struct
 * \ingroup group_rppdefs
 */
typedef struct
{
    unsigned int x;
    unsigned int y;
    unsigned int roiWidth;
    unsigned int roiHeight;
} RppiROI;

/******************** RPPT typedefs ********************/

/*! \brief RPPT Tensor datatype enum
 * \ingroup group_rppdefs
 */
typedef enum
{
    U8,
    F32,
    F16,
    I8,
    // ADD TYPES FOR GPU FUSION
    U16,
    S16,
    F64,
    S32
} RpptDataType;

/*! \brief RPPT Tensor layout type enum
 * \ingroup group_rppdefs
 */
typedef enum
{
    NCHW,   // BatchSize-Channels-Height-Width
    NHWC,   // BatchSize-Height-Width-Channels
    NCDHW,  // BatchSize-Channels-Depth-Height-Width
    NDHWC,  // BatchSize-Depth-Height-Width-Channels
    NHW,    // BatchSize-Height-Width
    NFT,    // BatchSize-Frequency-Time -> Frequency Major used for Spectrogram / MelfilterBank
    NTF     // BatchSize-Time-Frequency -> Time Major used for Spectrogram / MelfilterBank
} RpptLayout;

/*! \brief RPPT Tensor 2D ROI type enum
 * \ingroup group_rppdefs
 */
typedef enum
{
    LTRB,    // Left-Top-Right-Bottom
    XYWH     // X-Y-Width-Height
} RpptRoiType;

/*! \brief RPPT Tensor 3D ROI type enum
 * \ingroup group_rppdefs
 */
typedef enum
{
    LTFRBB,    // Left-Top-Front-Right-Bottom-Back
    XYZWHD     // X-Y-Z-Width-Height-Depth
} RpptRoi3DType;

/*! \brief RPPT Tensor subpixel layout type enum
 * \ingroup group_rppdefs
 */
typedef enum
{
    RGBtype,
    BGRtype
} RpptSubpixelLayout;

/*! \brief RPPT Tensor interpolation type enum
 * \ingroup group_rppdefs
 */
typedef enum
{
    NEAREST_NEIGHBOR = 0,
    BILINEAR,
    BICUBIC,
    LANCZOS,
    GAUSSIAN,
    TRIANGULAR
} RpptInterpolationType;

/*! \brief RPPT Audio Border Type
 * \ingroup group_rppdefs
 */
typedef enum
{
    ZERO = 0,
    CLAMP,
    REFLECT
} RpptAudioBorderType;

/*! \brief RPPT Mel Scale Formula
 * \ingroup group_rppdefs
 */
typedef enum
{
    SLANEY = 0,  // Follows Slaney’s MATLAB Auditory Modelling Work behavior
    HTK,         // Follows O’Shaughnessy’s book formula, consistent with Hidden Markov Toolkit(HTK), m = 2595 * log10(1 + (f/700))
} RpptMelScaleFormula;

/*! \brief RPPT Tensor 2D ROI LTRB struct
 * \ingroup group_rppdefs
 */
typedef struct
{
    RppiPoint lt, rb;    // Left-Top point and Right-Bottom point

} RpptRoiLtrb;

/*! \brief RPPT Tensor Channel Offsets struct
 * \ingroup group_rppdefs
 */
typedef struct
{
    RppiPoint r;
    RppiPoint g;
    RppiPoint b;
} RpptChannelOffsets;

/*! \brief RPPT Tensor 3D ROI LTFRBB struct
 * \ingroup group_rppdefs
 */
typedef struct
{
    RppiPoint3D ltf, rbb; // Left-Top-Front point and Right-Bottom-Back point

} RpptRoiLtfrbb;

/*! \brief RPPT Tensor 2D ROI XYWH struct
 * \ingroup group_rppdefs
 */
typedef struct
{
    RppiPoint xy;
    int roiWidth, roiHeight;

} RpptRoiXywh;

/*! \brief RPPT Tensor 3D ROI XYZWHD struct
 * \ingroup group_rppdefs
 */
typedef struct
{
    RppiPoint3D xyz;
    int roiWidth, roiHeight, roiDepth;

} RpptRoiXyzwhd;

/*! \brief RPPT Tensor 2D ROI union
 * \ingroup group_rppdefs
 */
typedef union
{
    RpptRoiLtrb ltrbROI;    // ROI defined as Left-Top-Right-Bottom
    RpptRoiXywh xywhROI;    // ROI defined as X-Y-Width-Height

} RpptROI, *RpptROIPtr;

/*! \brief RPPT Tensor 3D ROI union
 * \ingroup group_rppdefs
 */
typedef union
{
    RpptRoiLtfrbb ltfrbbROI;    // ROI defined as Left-Top-Front-Right-Bottom-Back
    RpptRoiXyzwhd xyzwhdROI;    // ROI defined as X-Y-Z-Width-Height-Depth

} RpptROI3D, *RpptROI3DPtr;

/*! \brief RPPT Tensor strides type struct
 * \ingroup group_rppdefs
 */
typedef struct
{
    Rpp32u nStride;
    Rpp32u cStride;
    Rpp32u hStride;
    Rpp32u wStride;
} RpptStrides;

/*! \brief RPPT Tensor descriptor type struct
 * \ingroup group_rppdefs
 */
typedef struct
{
    RppSize_t numDims;
    Rpp32u offsetInBytes;
    RpptDataType dataType;
    Rpp32u n, c, h, w;
    RpptStrides strides;
    RpptLayout layout;
} RpptDesc, *RpptDescPtr;

/*! \brief RPPT Tensor Generic descriptor type struct
 * \ingroup group_rppdefs
 */
typedef struct
{
    RppSize_t numDims;
    Rpp32u offsetInBytes;
    RpptDataType dataType;
    Rpp32u dims[RPPT_MAX_DIMS];
    Rpp32u strides[RPPT_MAX_DIMS];
    RpptLayout layout;
} RpptGenericDesc, *RpptGenericDescPtr;

/*! \brief RPPT Tensor 8-bit uchar RGB type struct
 * \ingroup group_rppdefs
 */
typedef struct
{
    Rpp8u R;
    Rpp8u G;
    Rpp8u B;
} RpptRGB;

/*! \brief RPPT Tensor 32-bit float RGB type struct
 * \ingroup group_rppdefs
 */
typedef struct
{
    Rpp32f R;
    Rpp32f G;
    Rpp32f B;
} RpptFloatRGB;

/*! \brief RPPT Tensor 2D 32-bit uint vector type struct
 * \ingroup group_rppdefs
 */
typedef struct
{
    Rpp32u x;
    Rpp32u y;
} RpptUintVector2D;

/*! \brief RPPT Tensor 2D 32-bit float vector type struct
 * \ingroup group_rppdefs
 */
typedef struct
{
    Rpp32f x;
    Rpp32f y;
} RpptFloatVector2D;

/*! \brief RPPT Tensor 2D image patch dimensions type struct
 * \ingroup group_rppdefs
 */
typedef struct
{
    Rpp32u width;
    Rpp32u height;
} RpptImagePatch, *RpptImagePatchPtr;

/*! \brief RPPT Tensor random number generator state (xorwow state) type struct
 * \ingroup group_rppdefs
 */
typedef struct
{   Rpp32u x[5];
    Rpp32u counter;
} RpptXorwowState;

/*! \brief RPPT Tensor random number generator state (xorwow box muller state) type struct
 * \ingroup group_rppdefs
 */
typedef struct
{   Rpp32s x[5];
    Rpp32s counter;
    int boxMullerFlag;
    float boxMullerExtra;
} RpptXorwowStateBoxMuller;

/*! \brief RPPT Tensor 2D bilinear neighborhood 32-bit signed int 8-length-vectors type struct
 * \ingroup group_rppdefs
 */
typedef struct
{
    Rpp32s24 srcLocsTL;
    Rpp32s24 srcLocsTR;
    Rpp32s24 srcLocsBL;
    Rpp32s24 srcLocsBR;
} RpptBilinearNbhoodLocsVecLen8;

/*! \brief RPPT Tensor 2D bilinear neighborhood 32-bit float 8-length-vectors type struct
 * \ingroup group_rppdefs
 */
typedef struct
{
    Rpp32f24 srcValsTL;
    Rpp32f24 srcValsTR;
    Rpp32f24 srcValsBL;
    Rpp32f24 srcValsBR;
} RpptBilinearNbhoodValsVecLen8;

/*! \brief RPPT Tensor GenericFilter type struct
 * \ingroup group_rppdefs
 */
typedef struct GenericFilter
{
    Rpp32f scale = 1.0f;
    Rpp32f radius = 1.0f;
    Rpp32s size;
    GenericFilter(RpptInterpolationType interpolationType, Rpp32s in_size, Rpp32s out_size, Rpp32f scaleRatio)
    {
        switch(interpolationType)
        {
        case RpptInterpolationType::BICUBIC:
        {
            this->radius = 2.0f;
            break;
        }
        case RpptInterpolationType::LANCZOS:
        {
            if(in_size > out_size)
            {
                this->radius = 3.0f * scaleRatio;
                this->scale = (1 / scaleRatio);
            }
            else
                this->radius = 3.0f;
            break;
        }
        case RpptInterpolationType::GAUSSIAN:
        {
            if(in_size > out_size)
            {
                this->radius = scaleRatio;
                this->scale = (1 / scaleRatio);
            }
            break;
        }
        case RpptInterpolationType::TRIANGULAR:
        {
            if(in_size > out_size)
            {
                this->radius = scaleRatio;
                this->scale = (1 / scaleRatio);
            }
            break;
        }
        default:
        {
            this->radius = 1.0f;
            this->scale = 1.0f;
            break;
        }
        }
        this->size = std::ceil(2 * this->radius);
    }
}GenericFilter;

/*! \brief RPPT Tensor RpptResamplingWindow type struct
 * \ingroup group_rppdefs
 */
typedef struct RpptResamplingWindow
{
    inline RPP_HOST_DEVICE void input_range(Rpp32f x, Rpp32s *loc0, Rpp32s *loc1)
    {
        Rpp32s xc = std::ceil(x);
        *loc0 = xc - lobes;
        *loc1 = xc + lobes;
    }

    inline Rpp32f operator()(Rpp32f x)
    {
        Rpp32f locRaw = x * scale + center;
        Rpp32s locFloor = std::floor(locRaw);
        Rpp32f weight = locRaw - locFloor;
        locFloor = std::max(std::min(locFloor, lookupSize - 2), 0);
        Rpp32f current = lookup[locFloor];
        Rpp32f next = lookup[locFloor + 1];
        return current + weight * (next - current);
    }

    inline __m128 operator()(__m128 x)
    {
        __m128 pLocRaw = _mm_add_ps(_mm_mul_ps(x, pScale), pCenter);
        __m128i pxLocFloor = _mm_cvttps_epi32(pLocRaw);
        __m128 pLocFloor = _mm_cvtepi32_ps(pxLocFloor);
        __m128 pWeight = _mm_sub_ps(pLocRaw, pLocFloor);
        Rpp32s idx[4];
        _mm_storeu_si128(reinterpret_cast<__m128i*>(idx), pxLocFloor);
        __m128 pCurrent = _mm_setr_ps(lookup[idx[0]], lookup[idx[1]], lookup[idx[2]], lookup[idx[3]]);
        __m128 pNext = _mm_setr_ps(lookup[idx[0] + 1], lookup[idx[1] + 1], lookup[idx[2] + 1], lookup[idx[3] + 1]);
        return _mm_add_ps(pCurrent, _mm_mul_ps(pWeight, _mm_sub_ps(pNext, pCurrent)));
    }

    Rpp32f scale = 1, center = 1;
    Rpp32s lobes = 0, coeffs = 0;
    Rpp32s lookupSize = 0;
    Rpp32f *lookupPinned = nullptr;
    std::vector<Rpp32f> lookup;
    __m128 pCenter, pScale;
} RpptResamplingWindow;

/*! \brief Base class for Mel scale conversions.
 * \ingroup group_rppdefs
 */
struct BaseMelScale
{
    public:
        inline RPP_HOST_DEVICE virtual Rpp32f hz_to_mel(Rpp32f hz) = 0;
        inline RPP_HOST_DEVICE virtual Rpp32f mel_to_hz(Rpp32f mel) = 0;
        virtual ~BaseMelScale() = default;
};

/*! \brief Derived class for HTK Mel scale conversions.
 * \ingroup group_rppdefs
 */
struct HtkMelScale : public BaseMelScale
{
    inline RPP_HOST_DEVICE Rpp32f hz_to_mel(Rpp32f hz) { return 1127.0f * std::log(1.0f + (hz / 700.0f)); }
    inline RPP_HOST_DEVICE Rpp32f mel_to_hz(Rpp32f mel) { return 700.0f * (std::exp(mel / 1127.0f) - 1.0f); }
    public:
        ~HtkMelScale() {};
};

/*! \brief Derived class for Slaney Mel scale conversions.
 * \ingroup group_rppdefs
 */
struct SlaneyMelScale : public BaseMelScale
{
    const Rpp32f freqLow = 0;
    const Rpp32f fsp = 66.666667f;
    const Rpp32f minLogHz = 1000.0;
    const Rpp32f minLogMel = (minLogHz - freqLow) / fsp;
    const Rpp32f stepLog = 0.068751777;  // Equivalent to std::log(6.4) / 27.0;

    const Rpp32f invMinLogHz = 0.001f;
    const Rpp32f invStepLog = 1.0f / stepLog;
    const Rpp32f invFsp = 1.0f / fsp;

    inline RPP_HOST_DEVICE Rpp32f hz_to_mel(Rpp32f hz)
    {
        Rpp32f mel = 0.0f;
        if (hz >= minLogHz)
            mel = minLogMel + std::log(hz * invMinLogHz) * invStepLog;
        else
            mel = (hz - freqLow) * invFsp;

        return mel;
    }

    inline RPP_HOST_DEVICE Rpp32f mel_to_hz(Rpp32f mel)
    {
        Rpp32f hz = 0.0f;
        if (mel >= minLogMel)
            hz = minLogHz * std::exp(stepLog * (mel - minLogMel));
        else
            hz = freqLow + mel * fsp;
        return hz;
    }
    public:
        ~SlaneyMelScale() {};
};

/******************** HOST memory typedefs ********************/

/*! \brief RPP HOST 32-bit float memory
 * \ingroup group_rppdefs
 */
typedef struct
{
    Rpp32f *floatmem;
} memRpp32f;

/*! \brief RPP HOST 64-bit double memory
 * \ingroup group_rppdefs
 */
typedef struct
{
    Rpp64f *doublemem;
} memRpp64f;

/*! \brief RPP HOST 32-bit unsigned int memory
 * \ingroup group_rppdefs
 */
typedef struct
{
    Rpp32u *uintmem;
} memRpp32u;

/*! \brief RPP HOST 32-bit signed int memory
 * \ingroup group_rppdefs
 */
typedef struct
{
    Rpp32s *intmem;
} memRpp32s;

/*! \brief RPP HOST 8-bit unsigned char memory
 * \ingroup group_rppdefs
 */
typedef struct
{
    Rpp8u *ucharmem;
} memRpp8u;

/*! \brief RPP HOST 8-bit signed char memory
 * \ingroup group_rppdefs
 */
typedef struct
{
    Rpp8s *charmem;
} memRpp8s;

/*! \brief RPP HOST RGB memory
 * \ingroup group_rppdefs
 */
typedef struct
{
    RpptRGB* rgbmem;
} memRpptRGB;

/*! \brief RPP HOST 2D dimensions memory
 * \ingroup group_rppdefs
 */
typedef struct
{
    Rpp32u *height;
    Rpp32u *width;
} memSize;

/*! \brief RPP HOST 2D ROI memory
 * \ingroup group_rppdefs
 */
typedef struct
{
    Rpp32u *x;
    Rpp32u *y;
    Rpp32u *roiHeight;
    Rpp32u *roiWidth;
} memROI;

/*! \brief RPP HOST memory type struct
 * \ingroup group_rppdefs
 */
typedef struct {
    RppiSize *srcSize;
    RppiSize *dstSize;
    RppiSize *maxSrcSize;
    RppiSize *maxDstSize;
    RppiROI *roiPoints;
    memRpp32f floatArr[10];
    memRpp64f doubleArr[10];
    memRpp32u uintArr[10];
    memRpp32s intArr[10];
    memRpp8u ucharArr[10];
    memRpp8s charArr[10];
    memRpptRGB rgbArr;
    Rpp64u *srcBatchIndex;
    Rpp64u *dstBatchIndex;
    Rpp32u *inc;
    Rpp32u *dstInc;
    Rpp32f *scratchBufferHost;
} memCPU;

#ifdef OCL_COMPILE

/******************** OCL memory typedefs ********************/

/*! \brief RPP OCL 32-bit float memory
 * \ingroup group_rppdefs
 */
typedef struct
{
    cl_mem floatmem;
} clmemRpp32f;

/*! \brief RPP OCL 64-bit double memory
 * \ingroup group_rppdefs
 */
typedef struct
{
    cl_mem doublemem;
} clmemRpp64f;

/*! \brief RPP OCL 32-bit unsigned int memory
 * \ingroup group_rppdefs
 */
typedef struct
{
    cl_mem uintmem;
} clmemRpp32u;

/*! \brief RPP OCL 32-bit signed int memory
 * \ingroup group_rppdefs
 */
typedef struct
{
    cl_mem intmem;
} clmemRpp32s;

/*! \brief RPP OCL 8-bit unsigned char memory
 * \ingroup group_rppdefs
 */
typedef struct
{
    cl_mem ucharmem;
} clmemRpp8u;

/*! \brief RPP OCL 8-bit signed char memory
 * \ingroup group_rppdefs
 */
typedef struct
{
    cl_mem charmem;
} clmemRpp8s;

/*! \brief RPP OCL 2D dimensions memory
 * \ingroup group_rppdefs
 */
typedef struct
{
    cl_mem height;
    cl_mem width;
} clmemSize;

/*! \brief RPP OCL 2D ROI memory
 * \ingroup group_rppdefs
 */
typedef struct
{
    cl_mem x;
    cl_mem y;
    cl_mem roiHeight;
    cl_mem roiWidth;
} clmemROI;

/*! \brief RPP OCL memory management type struct
 * \ingroup group_rppdefs
 */
typedef struct
{
    memSize csrcSize;
    memSize cdstSize;
    memSize cmaxSrcSize;
    memSize cmaxDstSize;
    memROI croiPoints;
    clmemSize srcSize;
    clmemSize dstSize;
    clmemSize maxSrcSize;
    clmemSize maxDstSize;
    clmemROI roiPoints;
    clmemRpp32f floatArr[10];
    clmemRpp64f doubleArr[10];
    clmemRpp32u uintArr[10];
    clmemRpp32s intArr[10];
    clmemRpp8u ucharArr[10];
    clmemRpp8s charArr[10];
    cl_mem srcBatchIndex;
    cl_mem dstBatchIndex;
    cl_mem inc;
    cl_mem dstInc;
} memGPU;

/*! \brief RPP OCL-HOST memory management
 * \ingroup group_rppdefs
 */
typedef struct
{
    memCPU mcpu;
    memGPU mgpu;
} memMgmt;

#elif defined(HIP_COMPILE)

/******************** HIP memory typedefs ********************/

/*! \brief RPP HIP 32-bit float memory
 * \ingroup group_rppdefs
 */
typedef struct
{
    Rpp32f* floatmem;
} hipMemRpp32f;

/*! \brief RPP HIP 64-bit double memory
 * \ingroup group_rppdefs
 */
typedef struct
{
    Rpp64f* doublemem;
} hipMemRpp64f;

/*! \brief RPP HIP 32-bit unsigned int memory
 * \ingroup group_rppdefs
 */
typedef struct
{
    Rpp32u* uintmem;
} hipMemRpp32u;

/*! \brief RPP HIP 32-bit signed int memory
 * \ingroup group_rppdefs
 */
typedef struct
{
    Rpp32s* intmem;
} hipMemRpp32s;

/*! \brief RPP HIP 8-bit unsigned char memory
 * \ingroup group_rppdefs
 */
typedef struct
{
    Rpp8u* ucharmem;
} hipMemRpp8u;

/*! \brief RPP HIP 8-bit signed char memory
 * \ingroup group_rppdefs
 */
typedef struct
{
    Rpp8s* charmem;
} hipMemRpp8s;

/*! \brief RPP HIP RGB memory
 * \ingroup group_rppdefs
 */
typedef struct
{
    RpptRGB* rgbmem;
} hipMemRpptRGB;

/*! \brief RPP HIP 2D dimensions memory
 * \ingroup group_rppdefs
 */
typedef struct
{
    Rpp32u* height;
    Rpp32u* width;
} hipMemSize;

/*! \brief RPP HIP 2D ROI memory
 * \ingroup group_rppdefs
 */
typedef struct
{
    Rpp32u* x;
    Rpp32u* y;
    Rpp32u* roiHeight;
    Rpp32u* roiWidth;
} hipMemROI;

/*! \brief RPP OCL memory management type struct
 * \ingroup group_rppdefs
 */
typedef struct
{
    memSize csrcSize;
    memSize cdstSize;
    memSize cmaxSrcSize;
    memSize cmaxDstSize;
    memROI croiPoints;
    hipMemSize srcSize;
    hipMemSize dstSize;
    hipMemSize maxSrcSize;
    hipMemSize maxDstSize;
    hipMemROI roiPoints;
    hipMemRpp32f floatArr[10];
    hipMemRpp32f float3Arr[10];
    hipMemRpp64f doubleArr[10];
    hipMemRpp32u uintArr[10];
    hipMemRpp32s intArr[10];
    hipMemRpp8u ucharArr[10];
    hipMemRpp8s charArr[10];
    hipMemRpptRGB rgbArr;
    hipMemRpp32f scratchBufferHip;
    Rpp64u* srcBatchIndex;
    Rpp64u* dstBatchIndex;
    Rpp32u* inc;
    Rpp32u* dstInc;
    hipMemRpp32f scratchBufferPinned;
} memGPU;

/*! \brief RPP HIP-HOST memory management
 * \ingroup group_rppdefs
 */
typedef struct
{
    memCPU mcpu;
    memGPU mgpu;
} memMgmt;

#else

/*! \brief RPP HOST memory management
 * \ingroup group_rppdefs
 */
typedef struct
{
    memCPU mcpu;
} memMgmt;

#endif //BACKEND

/*! \brief RPP initialize handle
 * \ingroup group_rppdefs
 */
typedef struct
{
    RppPtr_t cpuHandle;
    Rpp32u nbatchSize;
    memMgmt mem;
} InitHandle;

#endif /* RPPDEFS_H */

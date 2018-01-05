package com.komputation.cuda.optimization

import jcuda.Pointer

interface CudaUpdateRule {

    fun denseUpdate(
        numberParameters: Int,
        pointerToParameters: Pointer,
        pointerToGradient: Pointer)

    fun sparseUpdate(
        numberParameters: Int,
        pointerToParameterIndices: Pointer,
        pointerToCounts: Pointer,
        pointerToParameters: Pointer,
        pointerToGradient: Pointer)


}
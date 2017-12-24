package com.komputation.cuda.optimization

import jcuda.Pointer

interface CudaUpdateRule {

    fun denseUpdate(
        count : Int,
        pointerToParameters: Pointer,
        pointerToGradient: Pointer)

    fun sparseUpdate(
        hashTableSize: Int,
        pointerToHashTable: Pointer,
        pointerToCounts: Pointer,
        pointerToParameters: Pointer,
        pointerToGradient: Pointer)


}
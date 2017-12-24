package com.komputation.cuda.layers.continuation.activation

import com.komputation.cuda.kernels.Kernel

class CudaSigmoid internal constructor(
    name : String? = null,
    numberRows : Int,
    numberColumns : Int,
    createForwardKernel: () -> Kernel,
    createBackwardKernel: () -> Kernel,
    warpSize : Int,
    maximumNumberThreadsPerBlock : Int) : BaseCudaEntrywise(name, numberRows, numberColumns, createForwardKernel, createBackwardKernel, warpSize, maximumNumberThreadsPerBlock)
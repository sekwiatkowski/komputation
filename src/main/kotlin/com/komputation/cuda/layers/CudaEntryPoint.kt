package com.komputation.cuda.layers

import com.komputation.cuda.CudaForwardResult
import com.komputation.cuda.CudaLayer
import jcuda.Pointer
import com.komputation.cuda.memory.InputMemory
import com.komputation.matrix.Matrix

interface CudaEntryPoint : CudaLayer, CudaForwardResult {

    fun forward(
        batchId : Int,
        batchSize : Int,
        batch: IntArray,
        inputs: Array<Matrix>,
        memory : InputMemory) : Pointer

    fun backward(chain : Pointer) : Pointer

}
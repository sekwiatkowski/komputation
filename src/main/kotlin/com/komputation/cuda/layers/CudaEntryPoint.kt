package com.komputation.cuda.layers

import com.komputation.cuda.CudaForwardResult
import com.komputation.cuda.CudaLayer
import com.komputation.cuda.memory.InputMemory
import com.komputation.matrix.Matrix
import jcuda.Pointer

interface CudaEntryPoint : CudaLayer, CudaForwardResult {

    fun forward(
        batchId : Int,
        batchSize : Int,
        batch: IntArray,
        inputs: Array<Matrix>,
        memory : InputMemory) : Pointer

    fun backward(
        batchId : Int,
        chain : Pointer,
        memory : InputMemory) : Pointer

}
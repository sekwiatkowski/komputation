package com.komputation.cuda.network

import jcuda.Pointer
import com.komputation.cuda.layers.CudaEntryPoint
import com.komputation.cuda.layers.CudaForwardLayer
import com.komputation.cuda.memory.InputMemory
import com.komputation.matrix.Matrix

class CudaForwardPropagator(
    private val entryPoint: CudaEntryPoint,
    private val layers : Array<CudaForwardLayer>) {

    fun forward(batchId: Int, batchSize: Int, indices: IntArray, inputs: Array<Matrix>, memory : InputMemory, isTraining: Boolean) : Pointer {
        var result = this.entryPoint.forward(batchId, batchSize, indices, inputs, memory)

        for (layer in this.layers) {

            result = layer.forward(batchSize, result, isTraining)

        }

        return result
    }

}
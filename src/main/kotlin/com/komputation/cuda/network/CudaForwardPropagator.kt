package com.komputation.cuda.network

import com.komputation.cuda.CudaForwardResult
import jcuda.Pointer
import com.komputation.cuda.layers.CudaEntryPoint
import com.komputation.cuda.layers.CudaContinuation
import com.komputation.cuda.memory.InputMemory
import com.komputation.matrix.Matrix

class CudaForwardPropagator(
    private val entryPoint: CudaEntryPoint,
    private val layers : Array<CudaContinuation>) {

    fun forward(batchId: Int, batchSize: Int, indices: IntArray, inputs: Array<Matrix>, memory : InputMemory, isTraining: Boolean) : Pointer {
        this.entryPoint.forward(batchId, batchSize, indices, inputs, memory)

        var currentResult : CudaForwardResult = this.entryPoint

        for (layer in this.layers) {
            layer.forward(batchSize, currentResult.deviceForwardResult, currentResult.deviceForwardLengths, currentResult.batchMaximumOutputColumns, isTraining)

            currentResult = layer
        }

        return currentResult.deviceForwardResult
    }

}
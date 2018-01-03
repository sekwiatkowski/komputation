package com.komputation.cuda.network

import com.komputation.cuda.CudaForwardResult
import com.komputation.cuda.layers.CudaContinuation
import com.komputation.cuda.layers.CudaEntryPoint
import com.komputation.cuda.memory.InputMemory
import com.komputation.matrix.Matrix
import jcuda.Pointer

class CudaForwardPropagator(
    entryPoint: CudaEntryPoint,
    continuations: Array<CudaContinuation>) : BaseCudaPropagator(entryPoint, continuations) {

    fun forward(batchId: Int, batchSize: Int, indices: IntArray, inputs: Array<Matrix>, memory : InputMemory, isTraining: Boolean) : Pointer {
        val startEntry = System.nanoTime()
        this.entryPoint.forward(batchId, batchSize, indices, inputs, memory)
        val stopEntry = System.nanoTime()
        this.times[0] += stopEntry - startEntry

        var currentResult : CudaForwardResult = this.entryPoint

        for ((index, continuation) in this.continuations.withIndex()) {
            val startContinuation = System.nanoTime()
            continuation.forward(batchSize, currentResult.deviceForwardResult, currentResult.deviceForwardLengths, currentResult.largestNumberOutputColumnsInCurrentBatch, isTraining)
            val stopContinuation = System.nanoTime()
            this.times[index+1] += stopContinuation - startContinuation

            currentResult = continuation
        }

        return currentResult.deviceForwardResult
    }

}
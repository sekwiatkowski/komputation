package com.komputation.cuda.network

import jcuda.Pointer
import com.komputation.cuda.layers.CudaEntryPoint
import com.komputation.cuda.layers.CudaContinuation

class CudaBackwardPropagator(
    entryPoint: CudaEntryPoint,
    continuations: Array<CudaContinuation>) : BaseCudaPropagator(entryPoint, continuations) {

    fun backward(lossGradient : Pointer, batchSize: Int): Pointer {
        var chain = lossGradient

        for(indexLayer in this.numberContinuations - 1 downTo 0) {
            val continuation = this.continuations[indexLayer]

            val startContinuation = System.nanoTime()
            chain = continuation.backward(batchSize, chain)
            val stopContinuation = System.nanoTime()
            this.times[indexLayer+1] += stopContinuation - startContinuation
        }

        val startEntry = System.nanoTime()
        val backwardInput = this.entryPoint.backward(chain)
        val stopEntry = System.nanoTime()
        this.times[0] += stopEntry - startEntry

        return backwardInput
    }

}
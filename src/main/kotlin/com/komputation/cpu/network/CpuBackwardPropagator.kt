package com.komputation.cpu.network

import com.komputation.cpu.layers.CpuEntryPoint
import com.komputation.cpu.layers.CpuContinuation

class CpuBackwardPropagator(
    entryPoint: CpuEntryPoint,
    continuations: Array<CpuContinuation>) : BaseCpuPropagator(entryPoint, continuations) {

    fun backward(withinBatch: Int, lossGradient: FloatArray) : FloatArray {
        var chain = lossGradient

        for(indexLayer in this.numberContinuations - 1 downTo 0) {
            val continuation = this.continuations[indexLayer]

            val startContinuation = System.nanoTime()
            chain = continuation.backward(withinBatch, chain)
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
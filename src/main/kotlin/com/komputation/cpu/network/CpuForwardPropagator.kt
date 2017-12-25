package com.komputation.cpu.network

import com.komputation.cpu.layers.CpuEntryPoint
import com.komputation.cpu.layers.CpuContinuation
import com.komputation.cpu.layers.CpuForwardResult
import com.komputation.matrix.Matrix

class CpuForwardPropagator(
    entryPoint: CpuEntryPoint,
    continuations: Array<CpuContinuation>) : BaseCpuPropagator(entryPoint, continuations) {

    fun forward(withinBatch : Int, input : Matrix, isTraining : Boolean) : CpuForwardResult {
        this.entryPoint.forward(input)

        val startEntry = System.nanoTime()
        var currentResult : CpuForwardResult = this.entryPoint
        val stopEntry = System.nanoTime()
        this.times[0] += stopEntry - startEntry

        for ((index, continuation) in this.continuations.withIndex()) {
            val startContinuation = System.nanoTime()
            continuation.forward(withinBatch, currentResult.numberOutputColumns, currentResult.forwardResult, isTraining)
            val stopContinuation = System.nanoTime()
            this.times[index+1] += stopContinuation - startContinuation

            currentResult = continuation
        }

        return currentResult
    }

}
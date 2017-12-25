package com.komputation.cuda.network

import com.komputation.cuda.layers.CudaContinuation
import com.komputation.cuda.layers.CudaEntryPoint
import java.util.*

abstract class BaseCudaPropagator(
    protected val entryPoint: CudaEntryPoint,
    protected val continuations: Array<CudaContinuation>) {

    protected val numberContinuations = this.continuations.size
    protected val numberLayers = this.numberContinuations + 1
    protected val times = LongArray(this.numberLayers)

    fun stopTimer() : List<Pair<String?, Long>> {
        val times = getTimes()

        resetTimes()

        return times
    }

    private fun resetTimes() {
        Arrays.fill(this.times, 0)
    }

    private fun getTimes(): List<Pair<String?, Long>> {
        val names = listOf(this.entryPoint.name).plus(this.continuations.map { continuation -> continuation.name })

        return names.toList().zip(this.times.map { it.toDouble().div(1_000_000).toLong() })
    }
}
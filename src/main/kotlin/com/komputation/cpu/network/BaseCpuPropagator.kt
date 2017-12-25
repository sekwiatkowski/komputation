package com.komputation.cpu.network

import com.komputation.cpu.layers.CpuContinuation
import com.komputation.cpu.layers.CpuEntryPoint
import java.util.*

abstract class BaseCpuPropagator(
    protected val entryPoint: CpuEntryPoint,
    protected val continuations: Array<CpuContinuation>) {

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
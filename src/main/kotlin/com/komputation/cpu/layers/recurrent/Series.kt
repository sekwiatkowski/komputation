package com.komputation.cpu.layers.recurrent

import com.komputation.cpu.layers.CpuForwardLayer
import com.komputation.layers.Resourceful

open class Series internal constructor(
    private val name : String?,
    private val steps: Array<CpuForwardLayer>) : Resourceful {

    override fun acquire(maximumBatchSize: Int) {
        this.steps
            .filterIsInstance<Resourceful>()
            .forEach { step ->
                step.acquire(maximumBatchSize)
            }
    }

    override fun release() {
        this.steps
            .filterIsInstance<Resourceful>()
            .forEach { step ->
                step.release()
            }
    }

    fun forwardStep(withinBatch : Int, step : Int, numberInputColumns : Int, input : FloatArray, isTraining : Boolean) =
        this.steps[step].forward(withinBatch, numberInputColumns, input, isTraining)

    fun backwardStep(withinBatch: Int, step: Int, chain: FloatArray) =
        this.steps[step].backward(withinBatch, chain)

    fun getForwardResult(step : Int) =
        this.steps[step].forwardResult
}
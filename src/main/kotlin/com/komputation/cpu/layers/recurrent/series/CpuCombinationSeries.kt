package com.komputation.cpu.layers.recurrent.series

import com.komputation.cpu.layers.CpuCombination

open class CpuCombinationSeries internal constructor(
    private val name : String?,
    private val steps: Array<CpuCombination>)  {

    fun forwardStep(step : Int, first : FloatArray, second : FloatArray, numberInputColumns : Int) =
        this.steps[step].forward(first, second, numberInputColumns)

    fun backwardFirstStep(step : Int, chain : FloatArray) =
        this.steps[step].backwardFirst(chain)

    fun backwardSecondStep(step : Int, chain : FloatArray) =
        this.steps[step].backwardFirst(chain)

}
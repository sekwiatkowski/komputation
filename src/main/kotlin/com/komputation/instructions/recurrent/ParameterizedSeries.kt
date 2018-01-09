package com.komputation.instructions.recurrent

import com.komputation.cpu.instructions.CpuParameterizedSeriesInstruction
import com.komputation.cpu.layers.recurrent.series.CpuParameterizedSeries
import com.komputation.cpu.optimization.DenseAccumulator
import com.komputation.instructions.ShareableContinuationInstruction
import com.komputation.instructions.continuation.BaseHigherOrderInstruction
import com.komputation.optimization.OptimizationInstruction

class ParameterizedSeries internal constructor(
    private val name : String?,
    private val sharedParameter: FloatArray,
    private val numberParameterRows : Int,
    private val numberParameterColumns : Int,
    private val steps : Array<ShareableContinuationInstruction>,
    private val optimization: OptimizationInstruction? = null) : BaseHigherOrderInstruction(), CpuParameterizedSeriesInstruction {

    override fun getLayers() = steps

    override fun buildForCpu(): CpuParameterizedSeries {
        val seriesAccumulator = DenseAccumulator(this.sharedParameter.size)
        val batchAccumulator = DenseAccumulator(this.sharedParameter.size)

        val series = CpuParameterizedSeries(
            this.name,
            Array(this.steps.size) { index -> this.steps[index].buildForCpu(this.sharedParameter, seriesAccumulator) },
            this.sharedParameter,
            seriesAccumulator,
            batchAccumulator,
            this.optimization?.buildForCpu()?.invoke(this.numberParameterRows, this.numberParameterColumns))

        return series
    }
}

fun parameterizedSeries(
    name : String?,
    sharedParameter: FloatArray,
    numberParameterRows : Int,
    numberParameterColumns : Int,
    steps : Array<ShareableContinuationInstruction>,
    optimization: OptimizationInstruction? = null) =

    ParameterizedSeries(
        name,
        sharedParameter,
        numberParameterRows,
        numberParameterColumns,
        steps,
        optimization)
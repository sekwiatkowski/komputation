package com.komputation.instructions.recurrent

import com.komputation.cpu.instructions.CpuParameterizedSeriesInstruction
import com.komputation.cpu.layers.recurrent.series.CpuParameterizedSeries
import com.komputation.cpu.optimization.DenseAccumulator
import com.komputation.instructions.continuation.BaseHigherOrderInstruction
import com.komputation.instructions.ShareableContinuationInstruction
import com.komputation.optimization.OptimizationInstruction

class ParameterizedSeries internal constructor(
    private val name : String?,
    private val createSharedParameter: () -> FloatArray,
    private val numberParameterRows : Int,
    private val numberParameterColumns : Int,
    private val steps : Array<ShareableContinuationInstruction>,
    private val optimization: OptimizationInstruction? = null) : BaseHigherOrderInstruction(), CpuParameterizedSeriesInstruction {

    override fun getLayers() = steps

    override fun buildForCpu(): CpuParameterizedSeries {
        val sharedParameter = this.createSharedParameter()

        val seriesAccumulator = DenseAccumulator(sharedParameter.size)
        val batchAccumulator = DenseAccumulator(sharedParameter.size)

        val series = CpuParameterizedSeries(
            this.name,
            Array(this.steps.size) { index -> this.steps[index].buildForCpu(sharedParameter, seriesAccumulator) },
            sharedParameter,
            seriesAccumulator,
            batchAccumulator,
            this.optimization?.buildForCpu()?.invoke(this.numberParameterRows, this.numberParameterColumns))

        return series
    }
}

fun parameterizedSeries(
    name : String?,
    createSharedParameter: () -> FloatArray,
    numberParameterRows : Int,
    numberParameterColumns : Int,
    steps : Array<ShareableContinuationInstruction>,
    optimization: OptimizationInstruction? = null) =

    ParameterizedSeries(
        name,
        createSharedParameter,
        numberParameterRows,
        numberParameterColumns,
        steps,
        optimization)
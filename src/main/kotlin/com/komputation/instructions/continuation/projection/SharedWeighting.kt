package com.komputation.instructions.continuation.projection

import com.komputation.cpu.layers.continuation.projection.CpuWeighting
import com.komputation.cpu.optimization.DenseAccumulator
import com.komputation.instructions.ShareableContinuationInstruction
import com.komputation.optimization.OptimizationInstruction

class SharedWeighting internal constructor(
    name : String?,
    numberOutputRows : Int,
    optimizationStrategy : OptimizationInstruction? = null) : BaseWeighting(name, numberOutputRows, optimizationStrategy), ShareableContinuationInstruction {

    override fun buildForCpu(parameter : FloatArray, accumulator : DenseAccumulator): CpuWeighting {
        val updateRule = this.optimizationStrategy?.buildForCpu()?.invoke(this.numberWeightRows, this.numberWeightColumns)
        return CpuWeighting(this.name, this.numberInputRows, this.minimumNumberInputColumns, this.maximumNumberInputColumns, this.numberWeightRows, parameter, accumulator, updateRule)
    }
}

fun sharedWeighting(
    outputDimension: Int,
    optimizationStrategy : OptimizationInstruction? = null) =

    sharedWeighting(null, outputDimension, optimizationStrategy)

fun sharedWeighting(
    name : String?,
    outputDimension: Int,
    optimizationStrategy : OptimizationInstruction? = null) =

    SharedWeighting(
        name,
        outputDimension,
        optimizationStrategy)

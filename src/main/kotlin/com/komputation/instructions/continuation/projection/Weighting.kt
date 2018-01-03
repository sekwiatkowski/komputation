package com.komputation.instructions.continuation.projection

import com.komputation.cpu.instructions.CpuContinuationInstruction
import com.komputation.cpu.layers.continuation.projection.CpuWeighting
import com.komputation.cpu.optimization.DenseAccumulator
import com.komputation.cuda.CudaContext
import com.komputation.cuda.instructions.CudaContinuationInstruction
import com.komputation.cuda.kernels.ArrayKernels
import com.komputation.cuda.layers.continuation.projection.CublasWeighting
import com.komputation.initialization.InitializationStrategy
import com.komputation.initialization.initializeWeights
import com.komputation.optimization.OptimizationInstruction
import jcuda.jcublas.cublasHandle

class Weighting internal constructor(
    name : String?,
    numberOutputRows : Int,
    private val weightInitializationStrategy: InitializationStrategy,
    optimizationStrategy : OptimizationInstruction? = null) : BaseWeighting(name, numberOutputRows, optimizationStrategy), CpuContinuationInstruction, CudaContinuationInstruction {

    private fun initialWeights() = initializeWeights(this.weightInitializationStrategy, this.numberWeightRows, this.numberWeightColumns, this.maximumNumberInputEntries)

    override fun buildForCpu(): CpuWeighting {
        val accumulator = DenseAccumulator(this.numberWeightRows * this.numberWeightColumns)
        val updateRule = this.optimizationStrategy?.buildForCpu()?.invoke(this.numberWeightRows, this.numberWeightColumns)

        return CpuWeighting(this.name, this.numberInputRows, this.minimumNumberInputColumns, this.maximumNumberInputColumns, this.numberWeightRows, this.initialWeights(), accumulator, updateRule)
    }

    override fun buildForCuda(context: CudaContext, cublasHandle : cublasHandle): CublasWeighting {
        val updateRule = this.optimizationStrategy?.buildForCuda(context)?.invoke(1, this.numberWeightRows, this.numberWeightColumns)

        return CublasWeighting(
            this.name,
            cublasHandle,
            { context.createKernel(ArrayKernels.replaceNaN()) },
            this.numberInputRows,
            this.minimumNumberInputColumns,
            this.maximumNumberInputColumns,
            this.numberOutputRows,
            this.initialWeights(),
            updateRule,
            context.numberMultiprocessors,
            context.maximumNumberOfResidentWarpsPerMultiprocessor,
            context.warpSize,
            context.maximumNumberOfThreadsPerBlock)
    }

}

fun weighting(
    outputDimension: Int,
    initializationStrategy: InitializationStrategy,
    optimizationStrategy : OptimizationInstruction? = null) =

    weighting(null, outputDimension, initializationStrategy, optimizationStrategy)

fun weighting(
    name : String?,
    outputDimension: Int,
    initializationStrategy: InitializationStrategy,
    optimizationStrategy : OptimizationInstruction? = null) =

    Weighting(
        name,
        outputDimension,
        initializationStrategy,
        optimizationStrategy)
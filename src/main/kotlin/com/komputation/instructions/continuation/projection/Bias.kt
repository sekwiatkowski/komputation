package com.komputation.instructions.continuation.projection

import com.komputation.cpu.instructions.CpuContinuationInstruction
import com.komputation.cpu.layers.continuation.projection.CpuBias
import com.komputation.cpu.optimization.DenseAccumulator
import com.komputation.cuda.CudaContext
import com.komputation.cuda.instructions.CudaContinuationInstruction
import com.komputation.cuda.kernels.ForwardKernels
import com.komputation.cuda.layers.continuation.projection.CudaBias
import com.komputation.initialization.InitializationStrategy
import com.komputation.initialization.initializeColumnVector
import com.komputation.instructions.continuation.BaseEntrywiseInstruction
import com.komputation.optimization.OptimizationInstruction
import jcuda.jcublas.cublasHandle

class Bias internal constructor(
    private val name : String?,
    private val initializationStrategy: InitializationStrategy,
    private val optimizationStrategy : OptimizationInstruction? = null) : BaseEntrywiseInstruction(), CpuContinuationInstruction, CudaContinuationInstruction {

    private fun initializeBias() = initializeColumnVector(this.initializationStrategy, this.numberInputRows)

    override fun buildForCpu(): CpuBias {
        val accumulator = DenseAccumulator(this.numberInputRows)
        val updateRule = this.optimizationStrategy?.buildForCpu()?.invoke(this.numberInputRows, 1)

        val layer = CpuBias(this.name, this.numberInputRows, this.minimumNumberInputColumns, this.maximumNumberInputColumns, this.initializeBias(), accumulator, updateRule)

        return layer
    }

    override fun buildForCuda(context: CudaContext, cublasHandle: cublasHandle): CudaBias {
        val updateRule = this.optimizationStrategy?.buildForCuda(context)?.invoke(1, this.numberInputRows, 1)

        val layer = CudaBias(
            this.name,
            cublasHandle,
            this.numberInputRows,
            this.maximumNumberInputColumns,
            this.initializeBias(),
            updateRule,
            { context.createKernel(ForwardKernels.bias()) },
            context.warpSize,
            context.maximumNumberOfThreadsPerBlock)

        return layer
    }

}

fun bias(
    initializationStrategy: InitializationStrategy,
    optimizationStrategy : OptimizationInstruction? = null) =
    bias(null, initializationStrategy, optimizationStrategy)

fun bias(
    name : String?,
    initializationStrategy: InitializationStrategy,
    optimizationStrategy : OptimizationInstruction? = null) =
    Bias(name, initializationStrategy, optimizationStrategy)
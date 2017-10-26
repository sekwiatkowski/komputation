package com.komputation.optimization.adaptive

import com.komputation.cpu.optimization.adaptive.CpuAdagrad
import com.komputation.cuda.CudaContext
import com.komputation.cuda.kernels.OptimizationKernels
import com.komputation.cuda.optimization.adaptive.CudaAdagrad
import com.komputation.optimization.OptimizationInstruction

fun adagrad(learningRate: Float, epsilon: Float = 1e-6f) =

    Adagrad(learningRate, epsilon)

class Adagrad(private val learningRate: Float, private val epsilon: Float) : OptimizationInstruction {

    override fun buildForCpu() =

        { numberRows : Int, numberColumns : Int ->

            CpuAdagrad(this.learningRate, this.epsilon, numberRows * numberColumns)

        }

    override fun buildForCuda(context: CudaContext) =

        { numberParameters : Int, numberRows: Int, numberColumns: Int ->

            CudaAdagrad(
                numberParameters,
                numberRows * numberColumns,
                this.learningRate,
                this.epsilon,
                { context.createKernel(OptimizationKernels.adagrad()) },
                context.numberMultiprocessors,
                context.maximumNumberOfResidentWarpsPerMultiprocessor,
                context.warpSize,
                context.maximumNumberOfThreadsPerBlock)

        }

}
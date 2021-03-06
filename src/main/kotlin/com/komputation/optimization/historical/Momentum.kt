package com.komputation.optimization.historical

import com.komputation.cpu.optimization.historical.CpuMomentum
import com.komputation.cuda.CudaContext
import com.komputation.cuda.kernels.OptimizationKernels
import com.komputation.cuda.optimization.historical.CudaMomentum
import com.komputation.optimization.OptimizationInstruction

class Momentum internal constructor(private val learningRate: Float, private val momentum : Float) : OptimizationInstruction {

    override fun buildForCpu() =

        { numberRows : Int, numberColumns : Int ->

            CpuMomentum(this.learningRate, this.momentum, numberRows * numberColumns)

        }

    override fun buildForCuda(context: CudaContext) =

        { numberParameters : Int, numberRows : Int, numberColumns : Int ->

            CudaMomentum(
                numberParameters,
                numberRows * numberColumns,
                this.learningRate,
                this.momentum,
                { context.createKernel(OptimizationKernels.momentum()) },
                context.numberMultiprocessors,
                context.maximumNumberOfResidentWarpsPerMultiprocessor,
                context.warpSize,
                context.maximumNumberOfThreadsPerBlock)

        }

}

fun momentum(learningRate: Float, momentum : Float) =

    Momentum(learningRate, momentum)
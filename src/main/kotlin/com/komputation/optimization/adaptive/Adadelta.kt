package com.komputation.optimization.adaptive

import com.komputation.cpu.optimization.adaptive.CpuAdadelta
import com.komputation.cuda.CudaContext
import com.komputation.cuda.kernels.OptimizationKernels
import com.komputation.cuda.optimization.adaptive.CudaAdadelta
import com.komputation.optimization.OptimizationInstruction

fun adadelta(decay : Float = 0.95f, epsilon: Float = 1e-6f) =

    Adadelta(decay, epsilon)

class Adadelta(private val decay : Float, private val epsilon: Float) : OptimizationInstruction {

    override fun buildForCpu() =

        { numberRows : Int, numberColumns : Int ->

            CpuAdadelta(this.decay, this.epsilon, numberRows * numberColumns)

        }

    override fun buildForCuda(context: CudaContext) =

        { numberParameters : Int, numberRows: Int, numberColumns: Int ->

            CudaAdadelta(
                numberParameters,
                numberRows * numberColumns,
                this.decay,
                this.epsilon,
                { context.createKernel(OptimizationKernels.adadelta()) },
                context.numberMultiprocessors,
                context.maximumNumberOfResidentWarpsPerMultiprocessor,
                context.warpSize,
                context.maximumNumberOfThreadsPerBlock)

        }

}
package shape.komputation.optimization.historical

import shape.komputation.cpu.optimization.CpuOptimizationStrategy
import shape.komputation.cpu.optimization.historical.CpuNesterov
import shape.komputation.cuda.CudaContext
import shape.komputation.cuda.kernels.OptimizationKernels
import shape.komputation.cuda.optimization.CudaOptimizationStrategy
import shape.komputation.cuda.optimization.history.CudaNesterov
import shape.komputation.optimization.OptimizationInstruction

fun nesterov(learningRate: Float, momentum : Float) =

    Nesterov(learningRate, momentum)

class Nesterov(private val learningRate: Float, private val momentum : Float) : OptimizationInstruction {

    override fun buildForCpu() : CpuOptimizationStrategy {

        return { numberRows : Int, numberColumns : Int ->

            CpuNesterov(this.learningRate, this.momentum, numberRows * numberColumns)

        }

    }

    override fun buildForCuda(context: CudaContext): CudaOptimizationStrategy {

        return { numberParameters : Int, numberRows : Int, numberColumns : Int ->

            CudaNesterov(
                numberParameters,
                numberRows * numberColumns,
                this.learningRate,
                this.momentum,
                { context.createKernel(OptimizationKernels.nesterov()) },
                context.numberMultiprocessors,
                context.maximumNumberOfResidentWarpsPerMultiprocessor,
                context.warpSize,
                context.maximumNumberOfThreadsPerBlock)

        }

    }

}
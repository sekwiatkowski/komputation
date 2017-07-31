package shape.komputation.optimization

import shape.komputation.cpu.optimization.CpuOptimizationStrategy
import shape.komputation.cpu.optimization.CpuStochasticGradientDescent
import shape.komputation.cuda.CudaContext
import shape.komputation.cuda.kernels.OptimizationKernels
import shape.komputation.cuda.optimization.CudaOptimizationStrategy
import shape.komputation.cuda.optimization.CudaStochasticGradientDescent

fun stochasticGradientDescent(learningRate: Float) =

    StochasticGradientDescent(learningRate)

class StochasticGradientDescent(private val learningRate: Float) : OptimizationInstruction {

    override fun buildForCpu() : CpuOptimizationStrategy {

        return { _: Int, _: Int ->

            CpuStochasticGradientDescent(this.learningRate)

        }

    }

    override fun buildForCuda(context: CudaContext): CudaOptimizationStrategy {

        return { _ : Int, numberRows : Int, numberColumns : Int ->

            CudaStochasticGradientDescent(
                numberRows * numberColumns,
                this.learningRate,
                { context.createKernel(OptimizationKernels.stochasticGradientDescent()) },
                context.numberMultiprocessors,
                context.maximumNumberOfResidentWarpsPerMultiprocessor,
                context.warpSize,
                context.maximumNumberOfThreadsPerBlock)

        }

    }

}
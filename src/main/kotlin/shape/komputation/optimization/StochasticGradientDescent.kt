package shape.komputation.optimization

import shape.komputation.cpu.optimization.CpuOptimizationStrategy
import shape.komputation.cpu.optimization.CpuStochasticGradientDescent
import shape.komputation.cuda.CudaContext
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

        return { numberRows : Int, numberColumns : Int ->

            CudaStochasticGradientDescent(
                { context.kernelFactory.stochasticGradientDescent() },
                context.maximumNumberThreadsPerBlock,
                numberRows * numberColumns,
                this.learningRate)

        }

    }

}
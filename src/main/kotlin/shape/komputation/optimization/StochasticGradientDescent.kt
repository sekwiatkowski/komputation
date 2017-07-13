package shape.komputation.optimization

import jcuda.jcublas.cublasHandle
import shape.komputation.cpu.optimization.CpuOptimizationStrategy
import shape.komputation.cpu.optimization.CpuStochasticGradientDescent
import shape.komputation.cuda.optimization.CublasStochasticGradientDescent
import shape.komputation.cuda.optimization.CudaOptimizationStrategy

fun stochasticGradientDescent(learningRate: Double) =

    StochasticGradientDescent(learningRate)

class StochasticGradientDescent(private val learningRate: Double) : OptimizationInstruction {

    override fun buildForCpu() : CpuOptimizationStrategy {

        return { _: Int, _: Int ->

            CpuStochasticGradientDescent(this.learningRate)

        }

    }

    override fun buildForCuda(): CudaOptimizationStrategy {

        return { cublasHandle : cublasHandle, numberRows : Int, numberColumns : Int ->

            CublasStochasticGradientDescent(cublasHandle, numberRows, numberColumns, learningRate)

        }

    }

}
package shape.komputation.optimization.historical

import shape.komputation.cpu.optimization.CpuOptimizationStrategy
import shape.komputation.cpu.optimization.historical.CpuMomentum
import shape.komputation.cuda.CudaContext
import shape.komputation.cuda.kernels.OptimizationKernels
import shape.komputation.cuda.optimization.CudaOptimizationStrategy
import shape.komputation.cuda.optimization.history.CudaMomentum
import shape.komputation.optimization.OptimizationInstruction

fun momentum(learningRate: Float, momentum : Float) =

    Momentum(learningRate, momentum)

class Momentum(private val learningRate: Float, private val momentum : Float) : OptimizationInstruction {

    override fun buildForCpu() : CpuOptimizationStrategy {

        return { numberRows : Int, numberColumns : Int ->

            CpuMomentum(this.learningRate, this.momentum, numberRows * numberColumns)

        }

    }

    override fun buildForCuda(context: CudaContext): CudaOptimizationStrategy {

        return { numberParameters : Int, numberRows : Int, numberColumns : Int ->

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

}
package shape.komputation.optimization.historical

import shape.komputation.cpu.optimization.CpuOptimizationStrategy
import shape.komputation.cpu.optimization.historical.CpuNesterov
import shape.komputation.cuda.CudaContext
import shape.komputation.cuda.optimization.CudaOptimizationStrategy
import shape.komputation.optimization.OptimizationInstruction

fun nesterov(learningRate: Double, momentum : Double) =

    Nesterov(learningRate, momentum)

class Nesterov(private val learningRate: Double, private val momentum : Double) : OptimizationInstruction {

    override fun buildForCpu() : CpuOptimizationStrategy {

        return { numberRows : Int, numberColumns : Int ->

            CpuNesterov(this.learningRate, this.momentum, numberRows * numberColumns)

        }

    }

    override fun buildForCuda(context: CudaContext): CudaOptimizationStrategy {

        throw NotImplementedError()

    }

}
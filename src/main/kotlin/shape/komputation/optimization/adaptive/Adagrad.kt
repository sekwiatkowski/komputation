package shape.komputation.optimization.adaptive

import shape.komputation.cpu.optimization.CpuOptimizationStrategy
import shape.komputation.cpu.optimization.adaptive.CpuAdagrad
import shape.komputation.cuda.optimization.CudaOptimizationStrategy
import shape.komputation.optimization.OptimizationInstruction

fun adagrad(learningRate: Double, epsilon: Double = 1e-6) =

    Adagrad(learningRate, epsilon)

class Adagrad(private val decay : Double = 0.95, private val epsilon: Double = 1e-6) : OptimizationInstruction {

    override fun buildForCpu() : CpuOptimizationStrategy {

        return { numberRows : Int, numberColumns : Int ->

            CpuAdagrad(this.decay, this.epsilon, numberRows * numberColumns)

        }

    }

    override fun buildForCuda(): CudaOptimizationStrategy {

        throw NotImplementedError()

    }

}
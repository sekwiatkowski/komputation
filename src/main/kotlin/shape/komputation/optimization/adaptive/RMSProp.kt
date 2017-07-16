package shape.komputation.optimization.adaptive

import shape.komputation.cpu.optimization.CpuOptimizationStrategy
import shape.komputation.cpu.optimization.adaptive.CpuRMSProp
import shape.komputation.cuda.CudaContext
import shape.komputation.cuda.optimization.CudaOptimizationStrategy
import shape.komputation.optimization.OptimizationInstruction

fun rmsprop(learningRate: Double, decay : Double = 0.9, epsilon: Double = 1e-6) =

    RMSProp(learningRate, decay, epsilon)

class RMSProp(private val learningRate: Double, private val decay : Double = 0.9, private val epsilon: Double = 1e-6) : OptimizationInstruction {

    override fun buildForCpu() : CpuOptimizationStrategy {

        return { numberRows : Int, numberColumns : Int ->

            CpuRMSProp(this.learningRate, this.decay, this.epsilon, numberRows * numberColumns)

        }

    }

    override fun buildForCuda(context: CudaContext): CudaOptimizationStrategy {

        throw NotImplementedError()

    }

}
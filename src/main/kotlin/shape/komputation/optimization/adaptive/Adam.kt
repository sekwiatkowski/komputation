package shape.komputation.optimization.adaptive

import shape.komputation.cpu.optimization.CpuOptimizationStrategy
import shape.komputation.cpu.optimization.adaptive.CpuAdam
import shape.komputation.cuda.CudaContext
import shape.komputation.cuda.optimization.CudaOptimizationStrategy
import shape.komputation.optimization.OptimizationInstruction

fun adam(learningRate : Double = 0.001, firstMomentDecay : Double = 0.9, secondMomentDecay : Double = 0.999, epsilon : Double = 1e-8) =

    Adam(learningRate, firstMomentDecay, secondMomentDecay, epsilon)

class Adam(
    private val learningRate : Double = 0.001,
    private val firstMomentDecay : Double = 0.9,
    private val secondMomentDecay : Double = 0.999,
    private val epsilon : Double = 1e-8) : OptimizationInstruction {

    override fun buildForCpu() : CpuOptimizationStrategy {

        return { numberRows : Int, numberColumns : Int ->

            CpuAdam(this.learningRate, this.firstMomentDecay, this.secondMomentDecay, this.epsilon, numberRows * numberColumns)

        }

    }

    override fun buildForCuda(context: CudaContext): CudaOptimizationStrategy {

        throw NotImplementedError()

    }

}
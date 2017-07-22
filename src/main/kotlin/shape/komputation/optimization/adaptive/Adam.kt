package shape.komputation.optimization.adaptive

import shape.komputation.cpu.optimization.CpuOptimizationStrategy
import shape.komputation.cpu.optimization.adaptive.CpuAdam
import shape.komputation.cuda.CudaContext
import shape.komputation.cuda.optimization.CudaOptimizationStrategy
import shape.komputation.optimization.OptimizationInstruction

fun adam(learningRate : Float = 0.001f, firstMomentDecay : Float = 0.9f, secondMomentDecay : Float = 0.999f, epsilon : Float = 1e-8f) =

    Adam(learningRate, firstMomentDecay, secondMomentDecay, epsilon)

class Adam(
    private val learningRate : Float,
    private val firstMomentDecay : Float,
    private val secondMomentDecay : Float,
    private val epsilon : Float) : OptimizationInstruction {

    override fun buildForCpu() : CpuOptimizationStrategy {

        return { numberRows : Int, numberColumns : Int ->

            CpuAdam(this.learningRate, this.firstMomentDecay, this.secondMomentDecay, this.epsilon, numberRows * numberColumns)

        }

    }

    override fun buildForCuda(context: CudaContext): CudaOptimizationStrategy {

        throw NotImplementedError()

    }

}
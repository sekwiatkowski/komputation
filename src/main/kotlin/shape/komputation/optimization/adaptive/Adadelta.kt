package shape.komputation.optimization.adaptive

import shape.komputation.cpu.optimization.CpuOptimizationStrategy
import shape.komputation.cpu.optimization.adaptive.CpuAdadelta
import shape.komputation.cuda.optimization.CudaOptimizationStrategy
import shape.komputation.optimization.OptimizationInstruction

fun adadelta(decay : Double = 0.95, epsilon: Double = 1e-6) =

    Adadelta(decay, epsilon)

class Adadelta(private val decay : Double = 0.95, private val epsilon: Double = 1e-6) : OptimizationInstruction {

    override fun buildForCpu() : CpuOptimizationStrategy {

        return { numberRows : Int, numberColumns : Int ->

            CpuAdadelta(this.decay, this.epsilon, numberRows * numberColumns)

        }

    }

    override fun buildForCuda(): CudaOptimizationStrategy {

        throw NotImplementedError()

    }

}
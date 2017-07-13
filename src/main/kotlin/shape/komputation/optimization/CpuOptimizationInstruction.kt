package shape.komputation.optimization

import shape.komputation.cpu.optimization.CpuOptimizationStrategy

interface CpuOptimizationInstruction {

    fun buildForCpu() : CpuOptimizationStrategy

}
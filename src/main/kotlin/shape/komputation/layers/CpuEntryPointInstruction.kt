package shape.komputation.layers

import shape.komputation.cpu.layers.CpuEntryPoint

interface CpuEntryPointInstruction {

    fun buildForCpu() : CpuEntryPoint

}
package shape.komputation.layers

import shape.komputation.cpu.CpuEntryPoint

interface CpuEntryPointInstruction {

    fun buildForCpu() : CpuEntryPoint

}
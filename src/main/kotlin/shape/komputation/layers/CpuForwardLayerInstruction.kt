package shape.komputation.layers

import shape.komputation.cpu.CpuForwardLayer

interface CpuForwardLayerInstruction {

    fun buildForCpu() : CpuForwardLayer

}
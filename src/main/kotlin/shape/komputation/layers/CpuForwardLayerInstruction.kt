package shape.komputation.layers

import shape.komputation.cpu.layers.CpuForwardLayer

interface CpuForwardLayerInstruction {

    fun buildForCpu() : CpuForwardLayer

}
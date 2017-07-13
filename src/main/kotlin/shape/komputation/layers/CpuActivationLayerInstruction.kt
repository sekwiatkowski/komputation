package shape.komputation.layers

import shape.komputation.cpu.forward.activation.CpuActivationLayer

interface CpuActivationLayerInstruction : CpuForwardLayerInstruction {

    override fun buildForCpu(): CpuActivationLayer

}
package shape.komputation.layers

import shape.komputation.cpu.layers.forward.dropout.DropoutCompliant

interface CpuDropoutCompliantInstruction : CpuActivationLayerInstruction {

    override fun buildForCpu() : DropoutCompliant

}
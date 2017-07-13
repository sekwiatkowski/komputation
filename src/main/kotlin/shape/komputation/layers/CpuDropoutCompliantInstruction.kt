package shape.komputation.layers

import shape.komputation.cpu.forward.dropout.DropoutCompliant

interface CpuDropoutCompliantInstruction : CpuActivationLayerInstruction {

    override fun buildForCpu() : DropoutCompliant

}
package shape.komputation.layers

import shape.komputation.layers.forward.dropout.DropoutCompliant

interface CpuForwardLayerInstruction {

    fun buildForCpu() : ForwardLayer

}

interface DropoutCompliantInstruction : CpuForwardLayerInstruction {

    override fun buildForCpu() : DropoutCompliant

}
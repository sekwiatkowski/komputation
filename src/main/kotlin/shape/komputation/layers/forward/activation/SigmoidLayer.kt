package shape.komputation.layers.forward.activation

import shape.komputation.cpu.forward.activation.CpuSigmoidLayer
import shape.komputation.layers.CpuActivationLayerInstruction

class SigmoidLayer(private val name : String?) : CpuActivationLayerInstruction {

    override fun buildForCpu() =

        CpuSigmoidLayer(this.name)

}

fun sigmoidLayer(name : String? = null) = SigmoidLayer(name)
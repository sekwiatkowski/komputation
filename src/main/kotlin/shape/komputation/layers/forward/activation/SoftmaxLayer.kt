package shape.komputation.layers.forward.activation

import shape.komputation.cpu.layers.forward.activation.CpuSoftmaxLayer
import shape.komputation.layers.CpuActivationLayerInstruction

class SoftmaxLayer(private val name : String?) : CpuActivationLayerInstruction {

    override fun buildForCpu() =

        CpuSoftmaxLayer(this.name)

}

fun softmaxLayer(name : String? = null) = SoftmaxLayer(name)
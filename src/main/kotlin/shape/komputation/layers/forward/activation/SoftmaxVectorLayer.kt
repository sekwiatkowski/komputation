package shape.komputation.layers.forward.activation

import shape.komputation.cpu.layers.forward.activation.CpuSoftmaxVectorLayer
import shape.komputation.layers.CpuForwardLayerInstruction

class SoftmaxVectorLayer(private val name : String?) : CpuForwardLayerInstruction {

    override fun buildForCpu() =

        CpuSoftmaxVectorLayer(this.name)

}

fun softmaxVectorLayer(name : String? = null) = SoftmaxVectorLayer(name)
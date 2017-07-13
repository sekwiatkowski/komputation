package shape.komputation.layers.forward.activation

import shape.komputation.cpu.layers.forward.activation.CpuTanhLayer
import shape.komputation.layers.CpuActivationLayerInstruction

class TanhLayer(private val name : String?) : CpuActivationLayerInstruction {

    override fun buildForCpu() =

        CpuTanhLayer(this.name)

}

fun tanhLayer(name : String? = null) = TanhLayer(name)
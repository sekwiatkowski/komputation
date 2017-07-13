package shape.komputation.layers.forward.activation

import shape.komputation.cpu.layers.forward.activation.CpuIdentityLayer
import shape.komputation.layers.CpuActivationLayerInstruction

class IdentityLayer(private val name : String?) : CpuActivationLayerInstruction {

    override fun buildForCpu() =

        CpuIdentityLayer(name)

}

fun identityLayer(name : String? = null) =

    IdentityLayer(name)
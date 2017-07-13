package shape.komputation.layers.forward

import shape.komputation.layers.CpuForwardLayerInstruction
import shape.komputation.layers.forward.activation.ActivationLayer
import shape.komputation.matrix.DoubleMatrix

class CpuIdentityLayer internal constructor(name : String? = null) : ActivationLayer(name) {

    override fun forward(input : DoubleMatrix, isTraining : Boolean) =

        input

    override fun backward(chain : DoubleMatrix) =

        chain

}

class IdentityLayer(private val name : String?) : CpuForwardLayerInstruction {

    override fun buildForCpu() =

        CpuIdentityLayer(name)

}

fun identityLayer(name : String? = null) =

    IdentityLayer(name)
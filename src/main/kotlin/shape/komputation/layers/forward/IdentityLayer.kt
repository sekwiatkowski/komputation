package shape.komputation.layers.forward

import shape.komputation.layers.forward.activation.ActivationLayer
import shape.komputation.matrix.DoubleMatrix

class IdentityLayer(name : String? = null) : ActivationLayer(name) {

    override fun forward(input : DoubleMatrix, isTraining : Boolean) =

        input

    override fun backward(chain : DoubleMatrix) =

        chain

}

fun createIdentityLayer(name : String? = null) =

    IdentityLayer(name)
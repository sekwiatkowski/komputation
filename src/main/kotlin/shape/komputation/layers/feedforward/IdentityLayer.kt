package shape.komputation.layers.feedforward

import shape.komputation.layers.feedforward.activation.ActivationLayer
import shape.komputation.matrix.DoubleMatrix

class IdentityLayer(name : String? = null) : ActivationLayer(name) {

    override fun forward(input : DoubleMatrix) =

        input

    override fun backward(chain : DoubleMatrix) =

        chain

}

fun createIdentityLayer(name : String? = null) =

    IdentityLayer(name)
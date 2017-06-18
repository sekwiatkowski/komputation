package shape.komputation.layers.feedforward.projection

import shape.komputation.layers.FeedForwardLayer
import shape.komputation.matrix.DoubleMatrix

class IdentityProjectionLayer(name : String? = null) : FeedForwardLayer(name) {

    override fun forward(input: DoubleMatrix): DoubleMatrix {

        return input

    }

    override fun backward(chain: DoubleMatrix): DoubleMatrix {

        return chain

    }

}

fun createIdentityProjectionLayer(name : String?) =

    IdentityProjectionLayer(name)
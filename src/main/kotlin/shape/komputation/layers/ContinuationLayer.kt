package shape.komputation.layers

import shape.komputation.matrix.DoubleMatrix

sealed class ContinuationLayer(private val name: String?) {

    abstract fun forward(input: DoubleMatrix) : DoubleMatrix

}

abstract class FeedForwardLayer(name : String? = null) : ContinuationLayer(name) {

    abstract fun backward(chain : DoubleMatrix) : DoubleMatrix

}

abstract class RecurrentLayer(name : String?) : ContinuationLayer(name) {

    abstract fun backward(chain: DoubleMatrix) : Pair<DoubleMatrix, DoubleMatrix>

}
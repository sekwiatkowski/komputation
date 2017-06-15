package shape.komputation.layers

import shape.komputation.matrix.RealMatrix

sealed class ContinuationLayer(private val name: String?) {

    abstract fun forward(input: RealMatrix) : RealMatrix

}

abstract class FeedForwardLayer(name : String? = null) : ContinuationLayer(name) {

    abstract fun backward(chain : RealMatrix) : RealMatrix

}

abstract class RecurrentLayer(name : String?) : ContinuationLayer(name) {

    abstract fun resetForward()

    abstract fun resetBackward()

    abstract fun backward(chain: RealMatrix) : Pair<RealMatrix, RealMatrix>

}
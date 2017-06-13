package shape.komputation.layers.continuation

import shape.komputation.matrix.RealMatrix

abstract class ContinuationLayer(protected val name : String? = null) {

    abstract fun forward(input: RealMatrix) : RealMatrix

    abstract fun backward(chain : RealMatrix) : RealMatrix

}
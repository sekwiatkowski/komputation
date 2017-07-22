package shape.komputation.cpu.layers

import shape.komputation.matrix.FloatMatrix

abstract class CombinationLayer(private val name: String?) {

    abstract fun forward(first: FloatMatrix, second: FloatMatrix) : FloatMatrix

    abstract fun backwardFirst(chain : FloatMatrix) : FloatMatrix

    abstract fun backwardSecond(chain : FloatMatrix) : FloatMatrix

}
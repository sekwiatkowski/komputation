package shape.komputation.layers.feedforward.units

import shape.komputation.matrix.DoubleMatrix

abstract class RecurrentUnit(val name : String?) {

    abstract fun forward(state: DoubleMatrix, input: DoubleMatrix) : DoubleMatrix

    abstract fun backward(chain : DoubleMatrix): Pair<DoubleMatrix, DoubleMatrix>

}
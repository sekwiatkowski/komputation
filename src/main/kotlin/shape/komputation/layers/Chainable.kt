package shape.komputation.layers

import shape.komputation.matrix.DoubleMatrix

interface Chainable {

    fun backward(chain : DoubleMatrix) : DoubleMatrix

}
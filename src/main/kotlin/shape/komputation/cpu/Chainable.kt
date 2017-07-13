package shape.komputation.cpu

import shape.komputation.matrix.DoubleMatrix

interface Chainable {

    fun backward(chain : DoubleMatrix) : DoubleMatrix

}
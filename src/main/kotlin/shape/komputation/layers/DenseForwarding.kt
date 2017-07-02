package shape.komputation.layers

import shape.komputation.matrix.DoubleMatrix

interface DenseForwarding {

    fun forward(input: DoubleMatrix, isTraining : Boolean): DoubleMatrix

}
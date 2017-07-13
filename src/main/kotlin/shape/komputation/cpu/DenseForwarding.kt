package shape.komputation.cpu

import shape.komputation.matrix.DoubleMatrix

interface DenseForwarding {

    fun forward(input: DoubleMatrix, isTraining : Boolean): DoubleMatrix

}
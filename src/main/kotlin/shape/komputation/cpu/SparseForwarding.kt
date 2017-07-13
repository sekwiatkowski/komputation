package shape.komputation.cpu

import shape.komputation.matrix.DoubleMatrix

interface SparseForwarding {

    fun forward(input: DoubleMatrix, mask: BooleanArray): DoubleMatrix

}
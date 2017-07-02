package shape.komputation.layers

import shape.komputation.matrix.DoubleMatrix

interface SparseForwarding {

    fun forward(input: DoubleMatrix, mask: BooleanArray): DoubleMatrix

}
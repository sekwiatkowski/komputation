package shape.komputation.cpu.layers

import shape.komputation.matrix.FloatMatrix

interface SparseForwarding {

    fun forward(input: FloatMatrix, mask: BooleanArray): FloatMatrix

}
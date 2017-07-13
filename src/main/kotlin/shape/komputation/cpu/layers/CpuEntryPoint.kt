package shape.komputation.cpu.layers

import shape.komputation.matrix.DoubleMatrix
import shape.komputation.matrix.Matrix

interface CpuEntryPoint {

    fun forward(input: Matrix) : DoubleMatrix

    fun backward(chain : DoubleMatrix) : DoubleMatrix

}
package shape.komputation.cpu.layers

import shape.komputation.matrix.FloatMatrix
import shape.komputation.matrix.Matrix

interface CpuEntryPoint {

    fun forward(input: Matrix) : FloatMatrix

    fun backward(chain : FloatMatrix) : FloatMatrix

}
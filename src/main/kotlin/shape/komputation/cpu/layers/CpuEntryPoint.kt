package shape.komputation.cpu.layers

import shape.komputation.matrix.Matrix

interface CpuEntryPoint : CpuForwardState {

    fun forward(input: Matrix) : FloatArray

    fun backward(chain : FloatArray) : FloatArray

}
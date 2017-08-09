package shape.komputation.cpu.layers

import shape.komputation.matrix.Matrix

interface CpuEntryPoint : ForwardLayerState {

    fun forward(input: Matrix) : FloatArray

    fun backward(chain : FloatArray)

}
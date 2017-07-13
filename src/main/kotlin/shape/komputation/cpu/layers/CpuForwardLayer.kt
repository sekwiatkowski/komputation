package shape.komputation.cpu.layers

import shape.komputation.matrix.DoubleMatrix

interface CpuForwardLayer {

    fun forward(input: DoubleMatrix, isTraining : Boolean): DoubleMatrix

    fun backward(chain : DoubleMatrix) : DoubleMatrix

}
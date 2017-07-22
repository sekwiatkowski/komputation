package shape.komputation.cpu.layers

import shape.komputation.matrix.FloatMatrix

interface CpuForwardLayer {

    fun forward(input: FloatMatrix, isTraining : Boolean): FloatMatrix

    fun backward(chain : FloatMatrix) : FloatMatrix

}
package shape.komputation.cpu.layers

import shape.komputation.matrix.FloatMatrix

interface CpuForwardLayer {

    fun forward(withinBatch : Int, input: FloatMatrix, isTraining : Boolean): FloatMatrix

    fun backward(withinBatch : Int, chain : FloatMatrix) : FloatMatrix

}
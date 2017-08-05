package shape.komputation.cpu.layers.forward.activation

import shape.komputation.matrix.FloatMatrix

class CpuIdentityLayer internal constructor(name : String? = null) : BaseCpuActivationLayer(name) {

    override fun forward(withinBatch : Int, input : FloatMatrix, isTraining : Boolean) =

        FloatMatrix(input.numberRows, input.numberColumns, input.entries)

    override fun backward(withinBatch : Int, chain : FloatMatrix) =

        chain

}
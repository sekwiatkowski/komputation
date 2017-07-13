package shape.komputation.cpu.layers.forward.activation

import shape.komputation.matrix.DoubleMatrix

class CpuIdentityLayer internal constructor(name : String? = null) : BaseCpuActivationLayer(name) {

    override fun forward(input : DoubleMatrix, isTraining : Boolean) =

        input

    override fun backward(chain : DoubleMatrix) =

        chain

}
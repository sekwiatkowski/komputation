package shape.komputation.cpu.layers.forward.activation

import shape.komputation.matrix.FloatMatrix

class CpuIdentityLayer internal constructor(name : String? = null) : BaseCpuActivationLayer(name) {

    override fun forward(input : FloatMatrix, isTraining : Boolean) =

        input

    override fun backward(chain : FloatMatrix) =

        chain

}
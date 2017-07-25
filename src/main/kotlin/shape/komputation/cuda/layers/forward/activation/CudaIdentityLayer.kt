package shape.komputation.cuda.layers.forward.activation

import jcuda.Pointer

class CudaIdentityLayer internal constructor(name : String? = null) : BaseCudaActivationLayer(name) {

    override fun forward(input : Pointer, isTraining : Boolean) =

        input

    override fun backward(chain : Pointer) =

        chain

}
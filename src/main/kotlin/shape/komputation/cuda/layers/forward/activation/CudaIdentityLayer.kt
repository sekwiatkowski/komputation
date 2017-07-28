package shape.komputation.cuda.layers.forward.activation

import jcuda.Pointer

class CudaIdentityLayer internal constructor(name : String? = null) : BaseCudaActivationLayer(name) {

    override fun forward(input : Pointer, batchSize : Int, isTraining : Boolean) =

        input

    override fun backward(chain : Pointer, batchSize : Int) =

        chain

}
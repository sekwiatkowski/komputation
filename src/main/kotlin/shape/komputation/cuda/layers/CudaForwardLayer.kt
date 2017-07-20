package shape.komputation.cuda.layers

import jcuda.Pointer
import shape.komputation.layers.Resourceful

interface CudaForwardLayer : Resourceful {

    fun forward(input : Pointer): Pointer

    fun backward(chain : Pointer) : Pointer

}
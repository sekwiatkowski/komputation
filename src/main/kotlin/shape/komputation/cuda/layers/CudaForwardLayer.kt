package shape.komputation.cuda.layers

import jcuda.Pointer

interface CudaForwardLayer {

    fun forward(input : Pointer): Pointer

    fun backward(chain : Pointer) : Pointer

}
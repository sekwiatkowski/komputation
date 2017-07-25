package shape.komputation.cuda.layers

import jcuda.Pointer

interface CudaForwardLayer {

    fun forward(input : Pointer, isTraining : Boolean): Pointer

    fun backward(chain : Pointer) : Pointer

}
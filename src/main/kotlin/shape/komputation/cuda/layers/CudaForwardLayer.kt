package shape.komputation.cuda.layers

import jcuda.Pointer

interface CudaForwardLayer {

    fun forward(input : Pointer, batchSize : Int, isTraining : Boolean): Pointer

    fun backward(chain : Pointer, batchSize : Int) : Pointer

}
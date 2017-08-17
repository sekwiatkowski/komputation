package shape.komputation.cuda.layers

import jcuda.Pointer
import shape.komputation.cuda.CudaLayerState

interface CudaForwardLayer : CudaLayerState {

    fun forward(batchSize: Int, numberInputColumns : Int, input: Pointer, isTraining: Boolean): Pointer

    fun backward(batchSize: Int, chain: Pointer) : Pointer

}
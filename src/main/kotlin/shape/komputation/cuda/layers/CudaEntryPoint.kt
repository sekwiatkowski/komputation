package shape.komputation.cuda.layers

import jcuda.Pointer
import shape.komputation.cuda.CudaForwardState
import shape.komputation.cuda.memory.InputMemory
import shape.komputation.matrix.Matrix

interface CudaPropagation {

    fun forward(
        batchId : Int,
        batchSize : Int,
        batch: IntArray,
        inputs: Array<Matrix>,
        memory : InputMemory) : Pointer

    fun backward(chain : Pointer) : Pointer

}

interface CudaEntryPoint : CudaForwardState, CudaPropagation
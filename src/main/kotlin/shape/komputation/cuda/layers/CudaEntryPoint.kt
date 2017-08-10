package shape.komputation.cuda.layers

import jcuda.Pointer
import shape.komputation.cuda.InputMemory
import shape.komputation.matrix.Matrix

interface CudaEntryPoint {

    fun forward(batchId : Int, batchSize : Int, batch: IntArray, inputs: Array<Matrix>, memory : InputMemory) : Pointer

    fun backward(chain : Pointer) : Pointer

}
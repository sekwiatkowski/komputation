package shape.komputation.cuda.layers

import jcuda.Pointer
import shape.komputation.matrix.Matrix

interface CudaEntryPoint {

    fun forward(batchId : Int, inputIndices: IntArray, batchSize : Int, inputs: Array<Matrix>) : Pointer

    fun backward(chain : Pointer) : Pointer

}
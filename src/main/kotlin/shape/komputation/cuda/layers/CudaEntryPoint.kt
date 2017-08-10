package shape.komputation.cuda.layers

import jcuda.Pointer
import shape.komputation.matrix.Matrix

interface CudaEntryPoint {

    fun forward(batchId : Int, batchSize : Int, inputIndices: IntArray, inputs: Array<Matrix>, memory : HashMap<Int, Pointer>) : Pointer

    fun backward(chain : Pointer) : Pointer

}
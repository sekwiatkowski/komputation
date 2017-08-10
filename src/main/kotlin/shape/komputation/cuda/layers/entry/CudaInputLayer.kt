package shape.komputation.cuda.layers.entry

import jcuda.Pointer
import shape.komputation.cuda.InputMemory
import shape.komputation.cuda.layers.BaseCudaEntryPoint
import shape.komputation.matrix.Matrix

class CudaInputLayer(name : String?) : BaseCudaEntryPoint(name) {

    override fun forward(batchId : Int, batchSize : Int, batch: IntArray, inputs : Array<Matrix>, memory: InputMemory) =

        memory.get(batchId, batch, inputs)

    override fun backward(chain: Pointer) =

        chain

}
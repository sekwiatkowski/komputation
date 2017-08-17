package shape.komputation.cuda.layers.entry

import jcuda.Pointer
import shape.komputation.cuda.InputMemory
import shape.komputation.cuda.layers.BaseCudaEntryPoint
import shape.komputation.matrix.Matrix

class CudaInputLayer(
    name : String?,
    numberInputRows: Int,
    numberInputColumns: Int) : BaseCudaEntryPoint(name) {

    override var deviceForwardResult = Pointer()
    override val numberOutputRows = numberInputRows
    override val numberOutputColumns = numberInputColumns

    override fun forward(batchId : Int, batchSize : Int, batch: IntArray, inputs : Array<Matrix>, memory: InputMemory): Pointer {

        this.deviceForwardResult = memory.get(batchId, batch, inputs)

        return this.deviceForwardResult

    }

    override fun backward(chain: Pointer) =

        chain

}
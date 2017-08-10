package shape.komputation.cuda.layers.entry

import jcuda.Pointer
import shape.komputation.cuda.layers.BaseCudaEntryPoint
import shape.komputation.cuda.setFloatArray
import shape.komputation.matrix.FloatMatrix
import shape.komputation.matrix.Matrix

class CudaInputLayer(name : String?, numberRows: Int, numberColumns: Int) : BaseCudaEntryPoint(name) {

    private val numberEntries = numberRows * numberColumns

    override fun forward(batchId : Int, batchSize : Int, inputIndices: IntArray, inputs: Array<Matrix>, memory : HashMap<Int, Pointer>) =

        if (memory.containsKey(batchId)) {

            memory[batchId]!!

        }
        else {

            val numberBatchEntries = this.numberEntries * batchSize
            val batchEntries = FloatArray(numberBatchEntries)

            for ((index, indexInput) in inputIndices.withIndex()) {

                val input = inputs[indexInput]

                input as FloatMatrix

                System.arraycopy(input.entries, 0, batchEntries, index * this.numberEntries, this.numberEntries)

            }

            val deviceInput = Pointer()
            setFloatArray(batchEntries, numberBatchEntries, deviceInput)

            memory[batchId] = deviceInput

            deviceInput

        }

    override fun backward(chain: Pointer) =

        chain

}
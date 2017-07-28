package shape.komputation.cuda.layers.entry

import jcuda.Pointer
import jcuda.runtime.JCuda.cudaFree
import shape.komputation.cuda.layers.CudaEntryPoint
import shape.komputation.cuda.setFloatArray
import shape.komputation.layers.Resourceful
import shape.komputation.matrix.FloatMatrix
import shape.komputation.matrix.Matrix

class CudaInputLayer(private val numberEntries: Int) : CudaEntryPoint, Resourceful {

    private val memory = hashMapOf<Int, Pointer>()

    override fun acquire(maximumBatchSize : Int) {

    }

    override fun forward(batchId : Int, inputIndices: IntArray, batchSize : Int, inputs: Array<Matrix>) =

        if (this.memory.containsKey(batchId)) {

            this.memory[batchId]!!

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

            this.memory[batchId] = deviceInput

            deviceInput

        }

    override fun release() {

        for (deviceInput in this.memory.values) {

            cudaFree(deviceInput)

        }

    }

    override fun backward(chain: Pointer) =

        chain

}
package shape.komputation.cuda

import jcuda.Pointer
import jcuda.runtime.JCuda.cudaFree
import shape.komputation.matrix.FloatMatrix
import shape.komputation.matrix.Matrix

class InputMemory {

    private val memory = hashMapOf<Int, Pointer>()

    fun get(batchId : Int, batch : IntArray, inputs : Array<Matrix>) =

        if (this.memory.containsKey(batchId)) {

            this.memory[batchId]!!

        }
        else {

            val numberBatchEntries = batch.sumBy { index -> (inputs[index] as FloatMatrix).entries.size }
            val batchEntries = FloatArray(numberBatchEntries)

            var currentPosition = 0

            for (index in batch) {

                val input = inputs[index]
                input as FloatMatrix

                val inputEntries = input.entries
                val numberInputEntries = inputEntries.size

                System.arraycopy(inputEntries, 0, batchEntries, currentPosition, numberInputEntries)

                currentPosition += numberInputEntries

            }

            val deviceInput = Pointer()
            setFloatArray(batchEntries, numberBatchEntries, deviceInput)

            this.memory[batchId] = deviceInput

            deviceInput

        }

    fun release() {

        this.memory.values.forEach { pointer ->

            cudaFree(pointer)

        }

    }
}
package shape.komputation.cuda.layers.entry

import jcuda.Pointer
import jcuda.runtime.JCuda.cudaFree
import shape.komputation.cuda.layers.CudaEntryPoint
import shape.komputation.cuda.setVector
import shape.komputation.layers.Resourceful
import shape.komputation.matrix.FloatMatrix
import shape.komputation.matrix.Matrix

class CudaInputLayer(private val dimension : Int) : CudaEntryPoint, Resourceful {

    private val memory = hashMapOf<Int, Pointer>()

    override fun acquire() {

    }

    override fun forward(id : Int, input: Matrix) =

        if (this.memory.containsKey(id)) {

            this.memory[id]!!

        }
        else {

            input as FloatMatrix

            val deviceInput = Pointer()

            setVector(input.entries, this.dimension, deviceInput)

            this.memory[id] = deviceInput

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
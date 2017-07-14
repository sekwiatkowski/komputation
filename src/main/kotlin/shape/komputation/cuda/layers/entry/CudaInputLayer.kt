package shape.komputation.cuda.layers.entry

import jcuda.Pointer
import jcuda.runtime.JCuda.cudaFree
import shape.komputation.cuda.copyFromHostToDevice
import shape.komputation.cuda.layers.CudaEntryPoint
import shape.komputation.layers.Resourceful
import shape.komputation.matrix.DoubleMatrix
import shape.komputation.matrix.Matrix

class CudaInputLayer(private val dimension : Int) : CudaEntryPoint, Resourceful {

    private val memory = hashMapOf<Int, Pointer>()

    override fun acquire() {

    }

    override fun forward(id : Int, input: Matrix): Pointer {

        if (this.memory.containsKey(id)) {

            return this.memory[id]!!

        }
        else {

            input as DoubleMatrix

            val deviceInput = Pointer()

            copyFromHostToDevice(input.entries, this.dimension, deviceInput)

            this.memory[id] = deviceInput

            return deviceInput

        }

    }

    override fun release() {

        for (deviceInput in this.memory.values) {

            cudaFree(deviceInput)

        }

    }

    override fun backward(chain: Pointer) =

        chain

}
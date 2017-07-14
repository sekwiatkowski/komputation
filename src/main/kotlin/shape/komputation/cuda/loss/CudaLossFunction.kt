package shape.komputation.cuda.loss

import jcuda.Pointer

interface CudaLossFunction {

    fun accumulate(predictions: Pointer, targets : Pointer)

    fun accessAccumulation() : Double

    fun reset()

    fun backward(predictions: Pointer, targets : Pointer): Pointer

}
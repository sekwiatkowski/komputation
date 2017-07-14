package shape.komputation.cuda.loss

import jcuda.Pointer

interface CudaLossFunction {

    fun forward(predictions: Pointer, targets : Pointer): Double

    fun backward(predictions: Pointer, targets : Pointer): Pointer

}
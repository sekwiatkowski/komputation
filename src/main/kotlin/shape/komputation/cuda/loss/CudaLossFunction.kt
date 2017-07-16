package shape.komputation.cuda.loss

import jcuda.Pointer

interface CudaLossFunction {

    fun accumulate(pointerToPredictions: Pointer, pointerToTargets : Pointer)

    fun accessAccumulation() : Double

    fun reset()

    fun backward(pointerToPredictions: Pointer, pointerToTargets : Pointer): Pointer

}
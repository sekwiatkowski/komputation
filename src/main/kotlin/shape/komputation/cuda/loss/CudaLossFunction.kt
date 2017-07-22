package shape.komputation.cuda.loss

import jcuda.Pointer
import shape.komputation.layers.Resourceful

interface CudaLossFunction : Resourceful {

    fun accumulate(pointerToPredictions: Pointer, pointerToTargets : Pointer)

    fun accessAccumulation() : Float

    fun reset()

    fun backward(pointerToPredictions: Pointer, pointerToTargets : Pointer): Pointer

}
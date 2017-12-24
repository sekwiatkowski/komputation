package com.komputation.cuda.loss

import jcuda.Pointer
import com.komputation.instructions.Resourceful

interface CudaLossFunction : Resourceful {

    fun accumulate(pointerToPredictions: Pointer, pointerToTargets : Pointer, batchSize: Int)

    fun accessAccumulation() : Float

    fun backward(batchSize: Int, pointerToPredictions: Pointer, pointerToTargets: Pointer): Pointer

}
package com.komputation.cuda.workflow

import com.komputation.instructions.Resourceful
import jcuda.Pointer

interface CudaClassificationTester : Resourceful {

    fun resetCount()

    fun computeAccuracy() : Float

    fun evaluateBatch(batchSize: Int, pointerToPredictions : Pointer, pointerToTargets : Pointer)

}
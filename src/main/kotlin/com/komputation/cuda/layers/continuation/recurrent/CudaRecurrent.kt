package com.komputation.cuda.layers.continuation.recurrent

import com.komputation.cuda.layers.continuation.BaseCudaHigherOrderContinuation
import com.komputation.cuda.layers.continuation.projection.CublasProjection
import com.komputation.cuda.layers.continuation.projection.CudaBias
import com.komputation.instructions.Resourceful
import com.komputation.instructions.continuation.activation.Activation
import jcuda.Pointer
import jcuda.runtime.JCuda.cudaFree

class CudaRecurrent(
    name : String?,
    private val maximumLength : Int,
    inputDimension : Int,
    hiddenDimension : Int,
    private val inputWeighting : CublasProjection,
    private val bias : CudaBias?,
    private val activation : Activation) : BaseCudaHigherOrderContinuation(name, inputWeighting, inputWeighting), Resourceful {

    override val deviceForwardResult: Pointer
        get() = this.deviceResult
    override val deviceBackwardResult: Pointer
        get() = this.inputWeighting.deviceBackwardResult

    private val deviceResult = Pointer()
    private val pointerToResult = Pointer.to(this.deviceResult)

    private val hiddenDimension = this.inputWeighting.numberOutputRows

    override fun acquire(maximumBatchSize: Int) {
        acquire(maximumBatchSize * this.hiddenDimension * this.maximumLength)
    }

    override fun release() {
        cudaFree(this.deviceResult)
    }

    override fun forward(batchSize: Int, deviceInput: Pointer, deviceInputLengths: Pointer, largestNumberInputColumnsInBatch: Int, isTraining: Boolean): Pointer {

        /*
            Project the input:
               a1 b1 NaN NaN | c1 d1 e1 f1
               a2 b1 NaN NaN | c2 d2 e2 f2
               a2 b1 NaN NaN | c3 d3 e3 f3
            w1
            w2
         */

        TODO("not implemented")
    }

    override fun backward(batchSize: Int, chain: Pointer): Pointer {
        TODO("not implemented")
    }

}
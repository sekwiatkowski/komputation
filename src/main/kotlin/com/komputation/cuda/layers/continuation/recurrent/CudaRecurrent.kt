package com.komputation.cuda.layers.continuation.recurrent

import com.komputation.cuda.layers.continuation.BaseCudaHigherOrderContinuation
import com.komputation.cuda.layers.continuation.projection.CublasProjection
import com.komputation.optimization.Optimizable
import jcuda.Pointer

class CudaRecurrent(
    name : String?,
    private val inputProjection : CublasProjection,
    private val recurrentUnit : CudaRecurrentUnit) : BaseCudaHigherOrderContinuation(name, inputProjection, recurrentUnit), Optimizable {


    override fun forward(batchSize: Int, deviceInput: Pointer, deviceInputLengths: Pointer, isTraining: Boolean): Pointer {
        val projectedInput = this.inputProjection.forward(batchSize, deviceInput, deviceInputLengths, isTraining)

        return recurrentUnit.forward(batchSize, projectedInput, this.inputProjection.deviceForwardLengths, isTraining)

    }

    override fun backward(batchSize: Int, chain: Pointer): Pointer {
        val backwardRecurrentUnit = this.recurrentUnit.backward(batchSize, chain)

        return this.inputProjection.backward(batchSize, backwardRecurrentUnit)
    }

    override fun optimize(batchSize: Int) {
        this.inputProjection.optimize(batchSize)

        this.recurrentUnit.optimize(batchSize)
    }
}
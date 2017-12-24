package com.komputation.cuda.network

import jcuda.Pointer
import com.komputation.cuda.layers.CudaEntryPoint
import com.komputation.cuda.layers.CudaContinuation

class CudaBackwardPropagator(
    private val entryPoint: CudaEntryPoint,
    private val layers : Array<CudaContinuation>) {

    private val numberLayers = this.layers.size

    fun backward(lossGradient : Pointer, batchSize: Int): Pointer {
        var chain = lossGradient

        for(indexLayer in this.numberLayers - 1 downTo 0) {
            val layer = this.layers[indexLayer]

            chain = layer.backward(batchSize, chain)
        }

        return this.entryPoint.backward(chain)
    }

}
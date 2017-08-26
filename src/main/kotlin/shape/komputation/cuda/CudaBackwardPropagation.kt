package shape.komputation.cuda

import jcuda.Pointer
import shape.komputation.cuda.layers.CudaEntryPoint
import shape.komputation.cuda.layers.CudaForwardLayer

class CudaBackwardPropagator(
    private val entryPoint: CudaEntryPoint,
    private val layers : Array<CudaForwardLayer>) {

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
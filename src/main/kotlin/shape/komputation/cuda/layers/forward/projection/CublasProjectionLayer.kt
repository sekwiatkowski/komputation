package shape.komputation.cuda.layers.forward.projection

import jcuda.Pointer
import shape.komputation.cuda.layers.BaseCudaForwardLayer
import shape.komputation.layers.Resourceful
import shape.komputation.optimization.Optimizable

class CublasProjectionLayer internal constructor(
    name: String?,
    private val weightingLayer: CublasWeightingLayer,
    private val biasLayer: CublasBiasLayer? = null) : BaseCudaForwardLayer(name), Optimizable, Resourceful {

    override fun acquire(maximumBatchSize : Int) {

        this.weightingLayer.acquire(maximumBatchSize)

        this.biasLayer?.acquire(maximumBatchSize)

    }

    override fun forward(input : Pointer, batchSize : Int, isTraining : Boolean): Pointer {

        val weighted = this.weightingLayer.forward(input, batchSize, isTraining)

        if (this.biasLayer == null) {

            return weighted

        }
        else {

            return this.biasLayer.forward(weighted, batchSize, isTraining)

        }

    }

    override fun backward(chain: Pointer, batchSize : Int): Pointer {

        val backwardWeighting = this.weightingLayer.backward(chain, batchSize)

        if (this.biasLayer != null) {

            this.biasLayer.backward(chain, batchSize)

        }

        return backwardWeighting

    }

    override fun optimize(scalingFactor: Float) {

        this.weightingLayer.optimize(scalingFactor)

        this.biasLayer?.optimize(scalingFactor)

    }

    override fun release() {

        this.weightingLayer.release()

        this.biasLayer?.release()

    }

}
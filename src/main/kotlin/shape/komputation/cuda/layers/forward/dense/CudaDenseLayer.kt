package shape.komputation.cuda.layers.forward.dense

import jcuda.Pointer
import shape.komputation.cuda.layers.BaseCudaForwardLayer
import shape.komputation.cuda.layers.forward.activation.CudaActivationLayer
import shape.komputation.cuda.layers.forward.projection.CublasProjectionLayer
import shape.komputation.layers.Resourceful
import shape.komputation.optimization.Optimizable

class CudaDenseLayer internal constructor(
    name: String?,
    private val projectionLayer: CublasProjectionLayer,
    private val activationLayer: CudaActivationLayer) : BaseCudaForwardLayer(name), Optimizable, Resourceful {

    override fun acquire(maximumBatchSize : Int) {

        this.projectionLayer.acquire(maximumBatchSize)

        if (this.activationLayer is Resourceful) {

            this.activationLayer.acquire(maximumBatchSize)

        }

    }

    override fun forward(input : Pointer, batchSize : Int, isTraining : Boolean): Pointer {

        val projected = this.projectionLayer.forward(input, batchSize, isTraining)

        val activated = this.activationLayer.forward(projected, batchSize, isTraining)

        return activated

    }

    override fun backward(chain: Pointer, batchSize: Int): Pointer {

        val backwardActivation = this.activationLayer.backward(chain, batchSize)

        val backwardProjection = this.projectionLayer.backward(backwardActivation, batchSize)

        return backwardProjection

    }

    override fun optimize(scalingFactor: Float) {

        this.projectionLayer.optimize(scalingFactor)

    }

    override fun release() {

        if (this.activationLayer is Resourceful) {

            this.activationLayer.release()

        }

        this.projectionLayer.release()

    }

}
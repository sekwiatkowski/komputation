package shape.komputation.cuda.layers.forward.dense

import jcuda.Pointer
import shape.komputation.cuda.layers.BaseCudaForwardLayer
import shape.komputation.cuda.layers.forward.activation.CudaActivationLayer
import shape.komputation.cuda.layers.forward.projection.CudaProjectionLayer
import shape.komputation.layers.Resourceful
import shape.komputation.optimization.Optimizable

class CudaDenseLayer internal constructor(
    name: String?,
    private val projectionLayer: CudaProjectionLayer,
    private val activationLayer: CudaActivationLayer) : BaseCudaForwardLayer(name), Optimizable, Resourceful {

    override fun acquire() {

        this.projectionLayer.acquire()

        if (this.activationLayer is Resourceful) {

            this.activationLayer.acquire()

        }

    }

    override fun forward(input : Pointer, isTraining : Boolean): Pointer {

        val projected = this.projectionLayer.forward(input, isTraining)

        val activated = this.activationLayer.forward(projected, isTraining)

        return activated

    }

    override fun backward(chain: Pointer): Pointer {

        val backwardActivation = this.activationLayer.backward(chain)

        val backwardProjection = this.projectionLayer.backward(backwardActivation)

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
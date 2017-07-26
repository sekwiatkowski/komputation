package shape.komputation.cuda.layers.forward.activation

import jcuda.Pointer
import shape.komputation.cuda.getFloatArray
import shape.komputation.cuda.layers.forward.CudaNormalizationLayer
import shape.komputation.layers.Resourceful

class CudaSoftmaxLayer internal constructor(
    name : String? = null,
    private val exponentiationLayer: CudaExponentiationLayer,
    private val normalizationLayer: CudaNormalizationLayer) : BaseCudaActivationLayer(name), Resourceful {

    override fun acquire() {

        this.exponentiationLayer.acquire()

        this.normalizationLayer.acquire()

    }

    override fun forward(input : Pointer, isTraining : Boolean): Pointer {

        val exponentiated = this.exponentiationLayer.forward(input, isTraining)

        val normalized = this.normalizationLayer.forward(exponentiated, isTraining)

        return normalized

    }

    override fun backward(chain : Pointer) : Pointer {

        val backwardNormalization = this.normalizationLayer.backward(chain)

        val backwardExponentiation = this.exponentiationLayer.backward(backwardNormalization)

        return backwardExponentiation

    }

    override fun release() {

        this.exponentiationLayer.release()

        this.normalizationLayer.release()

    }

}
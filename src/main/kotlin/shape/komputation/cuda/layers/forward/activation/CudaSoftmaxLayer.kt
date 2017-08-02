package shape.komputation.cuda.layers.forward.activation

import jcuda.Pointer
import shape.komputation.cuda.getFloatArray
import shape.komputation.cuda.layers.forward.CudaNormalizationLayer
import shape.komputation.layers.Resourceful

class CudaSoftmaxLayer internal constructor(
    name : String? = null,
    private val exponentiationLayer: CudaExponentiationLayer,
    private val normalizationLayer: CudaNormalizationLayer) : BaseCudaActivationLayer(name), Resourceful {

    override fun acquire(maximumBatchSize : Int) {

        this.exponentiationLayer.acquire(maximumBatchSize)

        this.normalizationLayer.acquire(maximumBatchSize)

    }

    var count = 0

    override fun forward(input : Pointer, batchSize : Int, isTraining : Boolean): Pointer {

        val exponentiated = this.exponentiationLayer.forward(input, batchSize, isTraining)

        val normalized = this.normalizationLayer.forward(exponentiated, batchSize, isTraining)
        count++
        if(count==2) {
            val res = getFloatArray(normalized, batchSize * 10)
            res.forEach { println(it) }
        }

        return normalized

    }

    override fun backward(chain : Pointer, batchSize : Int) : Pointer {

        val backwardNormalization = this.normalizationLayer.backward(chain, batchSize)

        val backwardExponentiation = this.exponentiationLayer.backward(backwardNormalization, batchSize)

        return backwardExponentiation

    }

    override fun release() {

        this.exponentiationLayer.release()

        this.normalizationLayer.release()

    }

}
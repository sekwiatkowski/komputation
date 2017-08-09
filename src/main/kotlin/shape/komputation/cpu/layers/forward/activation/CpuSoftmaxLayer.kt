package shape.komputation.cpu.layers.forward.activation

import shape.komputation.cpu.layers.BaseCpuForwardLayer
import shape.komputation.cpu.layers.forward.CpuNormalizationLayer
import shape.komputation.layers.Resourceful

class CpuSoftmaxLayer internal constructor(
    name : String? = null,
    private val exponentiationLayer: CpuExponentiationLayer,
    private val normalizationLayer: CpuNormalizationLayer) : BaseCpuForwardLayer(name), Resourceful, CpuActivationLayer {

    override val numberOutputRows = this.normalizationLayer.numberOutputRows
    override var numberOutputColumns = this.normalizationLayer.numberOutputColumns
    override var forwardResult = FloatArray(0)

    override val numberInputRows = this.exponentiationLayer.numberInputRows
    override var numberInputColumns = this.exponentiationLayer.numberInputColumns
    override var backwardResult = FloatArray(0)

    override fun acquire(maximumBatchSize: Int) {

        this.exponentiationLayer.acquire(maximumBatchSize)
        this.normalizationLayer.acquire(maximumBatchSize)

    }

    override fun release() {

        this.exponentiationLayer.release()
        this.normalizationLayer.release()

    }

    override fun forward(withinBatch : Int, numberInputColumns : Int, input : FloatArray, isTraining : Boolean): FloatArray {

        this.numberInputColumns = numberInputColumns

        this.exponentiationLayer.forward(withinBatch, numberInputColumns, input, isTraining)

        this.forwardResult = this.normalizationLayer.forward(withinBatch, this.exponentiationLayer.numberOutputColumns, this.exponentiationLayer.forwardResult, isTraining)

        this.numberOutputColumns = this.normalizationLayer.numberOutputColumns

        return this.forwardResult

    }

    override fun backward(withinBatch : Int, chain : FloatArray): FloatArray {

        this.normalizationLayer.backward(withinBatch, chain)

        this.backwardResult = this.exponentiationLayer.backward(withinBatch, this.normalizationLayer.backwardResult)

        return this.backwardResult

    }

}
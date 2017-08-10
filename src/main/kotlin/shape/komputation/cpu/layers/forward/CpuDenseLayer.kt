package shape.komputation.cpu.layers.forward

import shape.komputation.cpu.layers.BaseCpuForwardLayer
import shape.komputation.cpu.layers.forward.activation.CpuActivationLayer
import shape.komputation.cpu.layers.forward.projection.CpuProjectionLayer
import shape.komputation.layers.Resourceful
import shape.komputation.optimization.Optimizable

class CpuDenseLayer internal constructor(
    name : String?,
    private val projection : CpuProjectionLayer,
    private val activation: CpuActivationLayer) : BaseCpuForwardLayer(name), Resourceful, Optimizable {

    override val numberOutputRows = this.activation.numberOutputRows
    override var numberOutputColumns = -1
    override var forwardResult = FloatArray(0)

    override val numberInputRows = this.projection.numberInputRows
    override var numberInputColumns = -1
    override var backwardResult = FloatArray(0)

    override fun acquire(maximumBatchSize: Int) {

        this.projection.acquire(maximumBatchSize)

        if (this.activation is Resourceful) {

            this.activation.acquire(maximumBatchSize)

        }

    }

    override fun release() {

        this.projection.release()

        if (this.activation is Resourceful) {

            this.activation.release()

        }

    }

    override fun forward(withinBatch : Int, numberInputColumns : Int, input: FloatArray, isTraining : Boolean): FloatArray {

        this.numberInputColumns = numberInputColumns
        this.numberOutputColumns = numberInputColumns

        val projected = this.projection.forward(withinBatch, numberInputColumns, input, isTraining)

        this.forwardResult = this.activation.forward(withinBatch, numberInputColumns, projected, isTraining)

        return this.forwardResult

    }

    override fun backward(withinBatch : Int, chain: FloatArray): FloatArray {

        this.activation.backward(withinBatch, chain)

        this.backwardResult = this.projection.backward(withinBatch, this.activation.backwardResult)

        return this.backwardResult

    }

    override fun optimize(scalingFactor : Float) {

        this.projection.optimize(scalingFactor)

    }

}
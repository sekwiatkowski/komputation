package shape.komputation.cpu.layers.forward

import shape.komputation.cpu.layers.BaseCpuForwardLayer
import shape.komputation.cpu.layers.forward.activation.CpuActivationLayer
import shape.komputation.cpu.layers.forward.projection.CpuProjectionLayer
import shape.komputation.matrix.FloatMatrix
import shape.komputation.optimization.Optimizable

class CpuDenseLayer internal constructor(
    name : String?,
    private val projection : CpuProjectionLayer,
    private val activation: CpuActivationLayer) : BaseCpuForwardLayer(name), Optimizable {

    override fun forward(withinBatch : Int, input: FloatMatrix, isTraining : Boolean): FloatMatrix {

        val projected = this.projection.forward(withinBatch, input, isTraining)

        val activated = this.activation.forward(withinBatch, projected, isTraining)

        return activated
    }

    override fun backward(withinBatch : Int, chain: FloatMatrix): FloatMatrix {

        val backwardActivation = this.activation.backward(withinBatch, chain)

        val backwardProjection = this.projection.backward(withinBatch, backwardActivation)

        return backwardProjection

    }

    override fun optimize(scalingFactor : Float) {

        this.projection.optimize(scalingFactor)

    }

}
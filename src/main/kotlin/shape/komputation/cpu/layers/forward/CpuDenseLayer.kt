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

    override fun forward(input: FloatMatrix, isTraining : Boolean): FloatMatrix {

        val projected = this.projection.forward(input, isTraining)

        val activated = this.activation.forward(projected, isTraining)

        return activated
    }

    override fun backward(chain: FloatMatrix): FloatMatrix {

        val diffChainWrtActivation = this.activation.backward(chain)

        val diffActivationWrtProjection = this.projection.backward(diffChainWrtActivation)

        return diffActivationWrtProjection

    }

    override fun optimize(scalingFactor : Float) {

        this.projection.optimize(scalingFactor)

    }

}
package shape.komputation.cpu.layers.forward

import shape.komputation.cpu.layers.BaseForwardLayer
import shape.komputation.cpu.layers.forward.activation.CpuActivationLayer
import shape.komputation.cpu.layers.forward.projection.CpuProjectionLayer
import shape.komputation.matrix.DoubleMatrix
import shape.komputation.optimization.Optimizable

class CpuDenseLayer internal constructor(
    name : String?,
    private val projection : CpuProjectionLayer,
    private val activation: CpuActivationLayer) : BaseForwardLayer(name), Optimizable {

    override fun forward(input: DoubleMatrix, isTraining : Boolean): DoubleMatrix {

        val projected = this.projection.forward(input, isTraining)

        val activated = this.activation.forward(projected, isTraining)

        return activated
    }

    override fun backward(chain: DoubleMatrix): DoubleMatrix {

        val diffChainWrtActivation = this.activation.backward(chain)

        val diffActivationWrtProjection = this.projection.backward(diffChainWrtActivation)

        return diffActivationWrtProjection

    }

    override fun optimize(scalingFactor : Double) {

        this.projection.optimize(scalingFactor)

    }

}
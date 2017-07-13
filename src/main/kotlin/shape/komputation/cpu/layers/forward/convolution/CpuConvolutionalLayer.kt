package shape.komputation.cpu.layers.forward.convolution

import shape.komputation.cpu.layers.BaseForwardLayer
import shape.komputation.cpu.layers.forward.projection.CpuProjectionLayer
import shape.komputation.matrix.DoubleMatrix
import shape.komputation.optimization.Optimizable

class CpuConvolutionalLayer internal constructor(
    name : String? = null,
    private val expansionLayer: CpuExpansionLayer,
    private val projectionLayer: CpuProjectionLayer) : BaseForwardLayer(name), Optimizable {

    override fun forward(input : DoubleMatrix, isTraining : Boolean) : DoubleMatrix {

        val expansion = this.expansionLayer.forward(input, isTraining)

        val projection = this.projectionLayer.forward(expansion, isTraining)

        return projection

    }

    override fun backward(chain : DoubleMatrix) : DoubleMatrix {

        val backwardProjection = this.projectionLayer.backward(chain)

        val backwardExpansion = this.expansionLayer.backward(backwardProjection)

        return backwardExpansion

    }

    override fun optimize(scalingFactor : Double) {

        this.projectionLayer.optimize(scalingFactor)

    }

}
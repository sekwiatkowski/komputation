package shape.komputation.cpu.layers.forward.convolution

import shape.komputation.cpu.layers.BaseCpuForwardLayer
import shape.komputation.cpu.layers.forward.projection.CpuProjectionLayer
import shape.komputation.matrix.FloatMatrix
import shape.komputation.optimization.Optimizable

class CpuConvolutionalLayer internal constructor(
    name : String? = null,
    private val expansionLayer: CpuExpansionLayer,
    private val projectionLayer: CpuProjectionLayer) : BaseCpuForwardLayer(name), Optimizable {

    override fun forward(input : FloatMatrix, isTraining : Boolean) : FloatMatrix {

        val expansion = this.expansionLayer.forward(input, isTraining)

        val projection = this.projectionLayer.forward(expansion, isTraining)

        return projection

    }

    override fun backward(chain : FloatMatrix) : FloatMatrix {

        val backwardProjection = this.projectionLayer.backward(chain)

        val backwardExpansion = this.expansionLayer.backward(backwardProjection)

        return backwardExpansion

    }

    override fun optimize(scalingFactor : Float) {

        this.projectionLayer.optimize(scalingFactor)

    }

}
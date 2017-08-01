package shape.komputation.cpu.layers.forward.convolution

import shape.komputation.cpu.layers.BaseCpuForwardLayer
import shape.komputation.cpu.layers.forward.projection.CpuProjectionLayer
import shape.komputation.matrix.FloatMatrix
import shape.komputation.optimization.Optimizable

class CpuConvolutionalLayer internal constructor(
    name : String? = null,
    private val expansionLayer: CpuExpansionLayer,
    private val projectionLayer: CpuProjectionLayer) : BaseCpuForwardLayer(name), Optimizable {

    override fun forward(withinBatch : Int, input : FloatMatrix, isTraining : Boolean) : FloatMatrix {

        val expansion = this.expansionLayer.forward(withinBatch, input, isTraining)

        val projection = this.projectionLayer.forward(withinBatch, expansion, isTraining)

        return projection

    }

    override fun backward(withinBatch : Int, chain : FloatMatrix) : FloatMatrix {

        val backwardProjection = this.projectionLayer.backward(withinBatch, chain)

        val backwardExpansion = this.expansionLayer.backward(withinBatch, backwardProjection)

        return backwardExpansion

    }

    override fun optimize(scalingFactor : Float) {

        this.projectionLayer.optimize(scalingFactor)

    }

}
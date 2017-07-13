package shape.komputation.layers.forward.convolution

import shape.komputation.initialization.InitializationStrategy
import shape.komputation.layers.BaseForwardLayer
import shape.komputation.layers.CpuForwardLayerInstruction
import shape.komputation.layers.concatenateNames
import shape.komputation.layers.forward.projection.CpuProjectionLayer
import shape.komputation.layers.forward.projection.projectionLayer
import shape.komputation.matrix.DoubleMatrix
import shape.komputation.optimization.Optimizable
import shape.komputation.optimization.OptimizationStrategy

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

class ConvolutionalLayer(
    private val name : String?,
    private val numberFilters: Int,
    private val filterWidth: Int,
    private val filterHeight : Int,
    private val initializationStrategy : InitializationStrategy,
    private val optimizationStrategy : OptimizationStrategy? = null) : CpuForwardLayerInstruction {

    override fun buildForCpu(): CpuConvolutionalLayer {

        val expansionLayerName = concatenateNames(name, "expansion")
        val expansionLayer = expansionLayer(expansionLayerName, this.filterWidth, this.filterHeight).buildForCpu()

        val projectionLayerName = concatenateNames(name, "projection")
        val projectionLayer = projectionLayer(projectionLayerName, this.filterWidth * this.filterHeight, this.numberFilters, this.initializationStrategy, this.initializationStrategy, this.optimizationStrategy).buildForCpu()

        return CpuConvolutionalLayer(name, expansionLayer, projectionLayer)

    }


}

fun convolutionalLayer(
    numberFilters: Int,
    filterWidth: Int,
    filterHeight : Int,
    initializationStrategy : InitializationStrategy,
    optimizationStrategy : OptimizationStrategy? = null): ConvolutionalLayer {

    return convolutionalLayer(null, numberFilters, filterWidth, filterHeight, initializationStrategy, optimizationStrategy)

}

fun convolutionalLayer(
    name : String?,
    numberFilters: Int,
    filterWidth: Int,
    filterHeight : Int,
    initializationStrategy : InitializationStrategy,
    optimizationStrategy : OptimizationStrategy? = null) =

    ConvolutionalLayer(name, numberFilters, filterWidth, filterHeight, initializationStrategy, optimizationStrategy)
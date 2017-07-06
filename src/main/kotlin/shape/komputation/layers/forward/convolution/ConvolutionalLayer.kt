package shape.komputation.layers.forward.convolution

import shape.komputation.initialization.InitializationStrategy
import shape.komputation.layers.ForwardLayer
import shape.komputation.layers.concatenateNames
import shape.komputation.layers.forward.projection.ProjectionLayer
import shape.komputation.layers.forward.projection.projectionLayer
import shape.komputation.matrix.DoubleMatrix
import shape.komputation.optimization.Optimizable
import shape.komputation.optimization.OptimizationStrategy

class ConvolutionalLayer internal constructor(
    name : String? = null,
    private val expansionLayer: ExpansionLayer,
    private val projectionLayer: ProjectionLayer) : ForwardLayer(name), Optimizable {

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
    optimizationStrategy : OptimizationStrategy? = null): ConvolutionalLayer {

    val expansionLayerName = concatenateNames(name, "expansion")
    val expansionLayer = expansionLayer(expansionLayerName, filterWidth, filterHeight)

    val projectionLayerName = concatenateNames(name, "projection")
    val projectionLayer = projectionLayer(projectionLayerName, filterWidth * filterHeight, numberFilters, initializationStrategy, initializationStrategy, optimizationStrategy)

    return ConvolutionalLayer(name, expansionLayer, projectionLayer)

}
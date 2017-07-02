package shape.komputation.layers.forward.convolution

import shape.komputation.initialization.InitializationStrategy
import shape.komputation.layers.ForwardLayer
import shape.komputation.layers.concatenateNames
import shape.komputation.layers.forward.projection.ProjectionLayer
import shape.komputation.layers.forward.projection.createProjectionLayer
import shape.komputation.matrix.DoubleMatrix
import shape.komputation.optimization.Optimizable
import shape.komputation.optimization.OptimizationStrategy

class ConvolutionalLayer(
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

    override fun optimize() {

        this.projectionLayer.optimize()

    }

}

fun createConvolutionalLayer(
    numberFilters: Int,
    filterWidth: Int,
    filterHeight : Int,
    initializationStrategy : InitializationStrategy,
    optimizationStrategy : OptimizationStrategy? = null): ConvolutionalLayer {

    return createConvolutionalLayer(null, numberFilters, filterWidth, filterHeight, initializationStrategy, optimizationStrategy)

}

fun createConvolutionalLayer(
    name : String?,
    numberFilters: Int,
    filterWidth: Int,
    filterHeight : Int,
    initializationStrategy : InitializationStrategy,
    optimizationStrategy : OptimizationStrategy? = null): ConvolutionalLayer {

    val expansionLayerName = concatenateNames(name, "expansion")
    val expansionLayer = createExpansionLayer(expansionLayerName, filterWidth, filterHeight)

    val projectionLayerName = concatenateNames(name, "projection")
    val projectionLayer = createProjectionLayer(projectionLayerName, filterWidth * filterHeight, numberFilters, initializationStrategy, initializationStrategy, optimizationStrategy)

    return ConvolutionalLayer(name, expansionLayer, projectionLayer)

}
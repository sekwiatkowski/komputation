package shape.komputation.layers.feedforward.convolution

import shape.komputation.initialization.InitializationStrategy
import shape.komputation.layers.FeedForwardLayer
import shape.komputation.layers.OptimizableLayer
import shape.komputation.layers.feedforward.ProjectionLayer
import shape.komputation.layers.feedforward.createProjectionLayer
import shape.komputation.matrix.RealMatrix
import shape.komputation.optimization.OptimizationStrategy

class ConvolutionalLayer(
    name : String? = null,
    private val expansionLayer: ExpansionLayer,
    private val projectionLayer: ProjectionLayer) : FeedForwardLayer(name), OptimizableLayer {

    override fun forward(input : RealMatrix) : RealMatrix {

        val expansion = this.expansionLayer.forward(input)

        val projection = this.projectionLayer.forward(expansion)

        return projection

    }

    override fun backward(chain : RealMatrix) : RealMatrix {

        val backwardProjectionLayer = this.projectionLayer.backward(chain)

        val backwardExpansionLayer = this.expansionLayer.backward(backwardProjectionLayer)

        return backwardExpansionLayer

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

    val expansionLayerName = if(name == null) null else "$name-expansion"
    val expansionLayer = createExpansionLayer(expansionLayerName, filterWidth, filterHeight)

    val projectionLayerName = if(name == null) null else "$name-projection"
    val projectionLayer = createProjectionLayer(projectionLayerName, filterWidth * filterHeight, numberFilters, initializationStrategy, optimizationStrategy)

    return ConvolutionalLayer(name, expansionLayer, projectionLayer)

}
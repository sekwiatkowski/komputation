package shape.komputation.layers.continuation.convolution

import shape.komputation.layers.continuation.ContinuationLayer
import shape.komputation.layers.continuation.OptimizableContinuationLayer
import shape.komputation.layers.continuation.ProjectionLayer
import shape.komputation.layers.continuation.createProjectionLayer
import shape.komputation.matrix.RealMatrix
import shape.komputation.optimization.UpdateRule

class ConvolutionalLayer(
    name : String? = null,
    private val expansionLayer: ExpansionLayer,
    private val projectionLayer: ProjectionLayer) : ContinuationLayer(name), OptimizableContinuationLayer {

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
    initializationStrategy : () -> Double,
    optimizationStrategy : ((numberRows : Int, numberColumns : Int) -> UpdateRule)? = null): ConvolutionalLayer {

    return createConvolutionalLayer(null, numberFilters, filterWidth, filterHeight, initializationStrategy, optimizationStrategy)

}

fun createConvolutionalLayer(
    name : String?,
    numberFilters: Int,
    filterWidth: Int,
    filterHeight : Int,
    initializationStrategy : () -> Double,
    optimizationStrategy : ((numberRows : Int, numberColumns : Int) -> UpdateRule)? = null): ConvolutionalLayer {

    val expansionLayerName = if(name == null) null else "$name-expansion"
    val expansionLayer = createExpansionLayer(expansionLayerName, filterWidth, filterHeight)

    val projectionLayerName = if(name == null) null else "$name-projection"
    val projectionLayer = createProjectionLayer(projectionLayerName, filterWidth * filterHeight, numberFilters, initializationStrategy, optimizationStrategy)

    return ConvolutionalLayer(name, expansionLayer, projectionLayer)

}
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
    private val projectionLayer: ProjectionLayer) : ContinuationLayer(name, 2, 2), OptimizableContinuationLayer {

    override fun forward() {

        this.expansionLayer.setInput(this.lastInput!!)
        this.expansionLayer.forward()
        val expansion = this.expansionLayer.lastForwardResult[0]

        this.projectionLayer.setInput(expansion)
        this.projectionLayer.forward()
        val projection = this.projectionLayer.lastForwardResult[0]

        this.lastForwardResult[0] = expansion
        this.lastForwardResult[1] = projection

    }

    override fun backward(chain : RealMatrix) {

        this.projectionLayer.backward(chain)

        this.expansionLayer.backward(this.projectionLayer.lastBackwardResultWrtInput!!)

        this.lastBackwardResultWrtInput = this.expansionLayer.lastBackwardResultWrtInput!!

        this.lastBackwardResultWrtParameters[0] = this.projectionLayer.lastBackwardResultWrtParameters.first()
        this.lastBackwardResultWrtParameters[1] = this.projectionLayer.lastBackwardResultWrtParameters.last()

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
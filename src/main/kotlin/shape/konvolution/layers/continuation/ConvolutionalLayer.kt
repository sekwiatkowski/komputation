package shape.konvolution.layers.continuation

import shape.konvolution.matrix.RealMatrix
import shape.konvolution.optimization.UpdateRule

class ConvolutionalLayer(
    private val expansionLayer: ExpansionLayer,
    private val projectionLayer: ProjectionLayer) : ContinuationLayer(2, 2), OptimizableContinuationLayer {

    override fun forward() {

        this.expansionLayer.setInput(this.lastInput!!)
        this.expansionLayer.forward()
        val expansion = this.expansionLayer.lastForwardResult[0]!!

        this.projectionLayer.setInput(expansion)
        this.projectionLayer.forward()

        this.lastForwardResult[0] = expansion
        this.lastForwardResult[1] = this.projectionLayer.lastForwardResult[0]

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

    val expansionLayer = ExpansionLayer(filterWidth, filterHeight)

    val projectionLayer = createProjectionLayer(filterWidth * filterHeight, numberFilters, initializationStrategy, optimizationStrategy)

    return ConvolutionalLayer(expansionLayer, projectionLayer)

}
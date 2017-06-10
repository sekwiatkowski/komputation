package shape.konvolution.layers.continuation

import shape.konvolution.BackwardResult
import shape.konvolution.initializeMatrix
import shape.konvolution.initializeRowVector
import shape.konvolution.matrix.RealMatrix
import shape.konvolution.optimization.UpdateRule

/* class ConvolutionLayer(
    filterWidth: Int,
    filterHeight: Int,
    weights : RealMatrix,
    bias : RealMatrix? = null,
    updateRule: UpdateRule? = null) : ContinuationLayer, OptimizableContinuationLayer {

    private val expansionLayer = ExpansionLayer(filterWidth, filterHeight)
    private val projectionLayer = ProjectionLayer(weights, bias, updateRule)

    override fun forward(input: RealMatrix): Array<RealMatrix> {

        val expansion = this.expansionLayer.forward(input).single()

        val projection = this.projectionLayer.forward(expansion).single()

        return arrayOf(expansion, projection)

    }

    override fun backward(inputs: Array<RealMatrix>, outputs : Array<RealMatrix>, chain : RealMatrix): BackwardResult {

        val (expansion, projection) = outputs

        val projectionBackward = this.projectionLayer.backward(arrayOf(expansion), arrayOf(projection), chain)

        val expansionBackward = this.expansionLayer.backward(inputs, arrayOf(expansion), projectionBackward.input)

        return BackwardResult(expansionBackward.input, projectionBackward.parameter)

    }

    override fun optimize(gradients: Array<RealMatrix?>) {

        this.projectionLayer.optimize(gradients)

    }

}

fun createConvolutionLayer(
    numberFilters: Int,
    filterWidth: Int,
    filterHeight : Int,
    initializationStrategy : () -> Double,
    updateRule: UpdateRule? = null): ConvolutionLayer {

    val weights = initializeMatrix(initializationStrategy, numberFilters,filterWidth * filterHeight)
    val bias = initializeRowVector(initializationStrategy, numberFilters)

    return ConvolutionLayer(filterWidth, filterHeight, weights, bias, updateRule)

} */
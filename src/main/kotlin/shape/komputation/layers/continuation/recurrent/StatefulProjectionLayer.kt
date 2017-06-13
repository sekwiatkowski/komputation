package shape.komputation.layers.continuation.recurrent

import shape.komputation.initializeMatrix
import shape.komputation.initializeRowVector
import shape.komputation.layers.continuation.ContinuationLayer
import shape.komputation.layers.continuation.OptimizableContinuationLayer
import shape.komputation.layers.continuation.ProjectionLayer
import shape.komputation.layers.continuation.differentiateProjectionWrtBias
import shape.komputation.matrix.RealMatrix
import shape.komputation.matrix.createRealVector
import shape.komputation.optimization.OptimizationStrategy
import shape.komputation.optimization.UpdateRule
import shape.komputation.optimization.updateDensely

class StatefulProjectionLayer(
    name : String?,
    initialState : RealMatrix,
    private val stateProjectionLayer: ProjectionLayer,
    private val inputProjectionLayer: ProjectionLayer,
    private val bias: RealMatrix? = null,
    private val biasUpdateRule: UpdateRule? = null) : ContinuationLayer(name, 1, 3), OptimizableContinuationLayer {

    private var state = initialState

    // project(state) + project(input) + bias
    override fun forward() {

        stateProjectionLayer.setInput(state)
        stateProjectionLayer.forward()

        val stateProjection = stateProjectionLayer.lastForwardResult[0]

        inputProjectionLayer.setInput(this.lastInput!!)
        inputProjectionLayer.forward()

        val inputProjection = inputProjectionLayer.lastForwardResult[0]

        val sum = stateProjection.add(inputProjection)

        this.lastForwardResult[0] = if (bias == null) sum else sum.add(bias)

    }

    override fun backward(chain: RealMatrix) {

        // d [ state weights * state + input weights * input + bias ] / d state weights
        stateProjectionLayer.backward(chain)
        this.lastBackwardResultWrtParameters[0] = stateProjectionLayer.lastBackwardResultWrtParameters.first()

        // d [ state weights * state + input weights * input + bias ] / d input weights
        inputProjectionLayer.backward(chain)
        this.lastBackwardResultWrtParameters[1] = inputProjectionLayer.lastBackwardResultWrtParameters.first()

        if (bias != null) {

            // d [ state weights * state + input weights * input + bias ] / d bias
            this.lastBackwardResultWrtParameters[2] = differentiateProjectionWrtBias(bias.numberRows(), chain)

        }

        // d [ state weights * state + input weights * input + bias ] / d input weights = d input weights * input / d input
        this.lastBackwardResultWrtInput = this.inputProjectionLayer.lastBackwardResultWrtInput

    }

    override fun optimize() {

        this.stateProjectionLayer.optimize()

        this.inputProjectionLayer.optimize()

        if (bias != null && biasUpdateRule != null) {

            val biasGradient = this.lastBackwardResultWrtParameters.last()

            updateDensely(this.bias, biasGradient!!, biasUpdateRule)

        }

    }

}

fun createStatefulProjectionLayer(
    name : String? = null,
    inputDimension : Int,
    hiddenDimension : Int,
    initializationStrategy: () -> Double,
    optimizationStrategy : OptimizationStrategy? = null) : StatefulProjectionLayer {

    val initialState = createRealVector(hiddenDimension)
    val stateWeights = initializeMatrix(initializationStrategy, hiddenDimension, hiddenDimension)
    val stateProjectionName = if(name == null) null else "$name-state-projection"
    val stateWeightUpdateRule = if(optimizationStrategy != null) optimizationStrategy(hiddenDimension, hiddenDimension) else null
    val stateProjectionLayer = ProjectionLayer(stateProjectionName, stateWeights, null, stateWeightUpdateRule, null)

    val inputWeights = initializeMatrix(initializationStrategy, hiddenDimension, inputDimension)
    val inputProjectionName = if(name == null) null else "$name-input-projection"
    val inputWeightUpdateRule = if(optimizationStrategy != null) optimizationStrategy(hiddenDimension, hiddenDimension) else null
    val inputProjectionLayer = ProjectionLayer(inputProjectionName, inputWeights, null, inputWeightUpdateRule, null)

    val bias = initializeRowVector(initializationStrategy, hiddenDimension)
    val biasUpdateRule = if(optimizationStrategy != null) optimizationStrategy(hiddenDimension, 1) else null

    return StatefulProjectionLayer(name, initialState, stateProjectionLayer, inputProjectionLayer, bias, biasUpdateRule)
}

/* fun main(args: Array<String>) {

    val random = Random(1)

    val inputDimension = 2
    val hiddenDimension = 10

    val state = createRealMatrix(10, 1)

    val initialize = createUniformInitializer(random, -0.01, 0.01)

    val input = createRealVector(1.0, 2.0)

    inputProjection.setInput(input)
    inputProjection.forward()

    val lastForwardResult = inputProjection.lastForwardResult[0]

} */
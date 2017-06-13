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
    private val biasUpdateRule: UpdateRule? = null) : ContinuationLayer(name), OptimizableContinuationLayer {

    private var state = initialState

    private var forwardResult : RealMatrix? = null
    private var backpropagationWrtBias : RealMatrix? = null

    // project(state) + project(input) + bias
    override fun forward(input : RealMatrix) : RealMatrix {

        val stateProjection = stateProjectionLayer.forward(state)

        val inputProjection = inputProjectionLayer.forward(input)

        val sum = stateProjection.add(inputProjection)

        this.forwardResult = if (bias == null) sum else sum.add(bias)

        return this.forwardResult!!

    }

    override fun backward(chain: RealMatrix) : RealMatrix {

        // d [ state weights * state + input weights * input + bias ] / d state weights
        stateProjectionLayer.backward(chain)

        // d [ state weights * state + input weights * input + bias ] / d input weights
        val backpropagationWrtInput = inputProjectionLayer.backward(chain)

        if (bias != null) {

            // d [ state weights * state + input weights * input + bias ] / d bias
            this.backpropagationWrtBias = differentiateProjectionWrtBias(bias.numberRows(), chain)

        }

        // d [ state weights * state + input weights * input + bias ] / d input weights = d input weights * input / d input
        return backpropagationWrtInput

    }

    override fun optimize() {

        this.stateProjectionLayer.optimize()

        this.inputProjectionLayer.optimize()

        if (bias != null && biasUpdateRule != null) {

            updateDensely(this.bias, this.backpropagationWrtBias!!, biasUpdateRule)

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
package shape.komputation.layers.continuation.recurrent

import shape.komputation.initialization.InitializationStrategy
import shape.komputation.initialization.initializeMatrix
import shape.komputation.initialization.initializeRowVector
import shape.komputation.layers.continuation.ContinuationLayer
import shape.komputation.layers.continuation.OptimizableContinuationLayer
import shape.komputation.layers.continuation.ProjectionLayer
import shape.komputation.layers.continuation.activation.ActivationLayer
import shape.komputation.layers.continuation.differentiateProjectionWrtBias
import shape.komputation.matrix.RealMatrix
import shape.komputation.matrix.createRealVector
import shape.komputation.optimization.OptimizationStrategy
import shape.komputation.optimization.UpdateRule
import shape.komputation.optimization.updateDensely

class RecurrentCell(
    name : String?,
    private val initialState : RealMatrix,
    private val stateProjectionLayer: ProjectionLayer,
    private val inputProjectionLayer: ProjectionLayer,
    private val activationLayer: ActivationLayer,
    private val bias: RealMatrix? = null,
    private val biasUpdateRule: UpdateRule? = null) : ContinuationLayer(name), OptimizableContinuationLayer {

    private var state = initialState
    private var step = 0

    private var backpropagationWrtBias : RealMatrix? = null

    fun reset() {

        this.state = initialState
        this.step = 0
    }

    // project(state) + project(input) + bias
    override fun forward(input : RealMatrix) : RealMatrix {

        val stateProjection = stateProjectionLayer.forward(state)

        val inputProjection = inputProjectionLayer.forward(input)

        val sum = stateProjection.add(inputProjection)

        val preActivation = if (bias == null) sum else sum.add(bias)

        this.state = activationLayer.forward(preActivation)

        return this.state

    }

    override fun backward(chain: RealMatrix) : RealMatrix {

        val backpropagationActivation = this.activationLayer.backward(chain)

        // d [ state weights * state + input weights * input + bias ] / d state weights
        stateProjectionLayer.backward(backpropagationActivation)

        // d [ state weights * state + input weights * input + bias ] / d input weights
        val backpropagationWrtInput = inputProjectionLayer.backward(backpropagationActivation)

        if (bias != null) {

            // d [ state weights * state + input weights * input + bias ] / d bias
            this.backpropagationWrtBias = differentiateProjectionWrtBias(bias.numberRows(), backpropagationActivation)

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
    activationLayer: ActivationLayer,
    initializationStrategy: InitializationStrategy,
    optimizationStrategy : OptimizationStrategy? = null) : RecurrentCell {

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

    return RecurrentCell(name, initialState, stateProjectionLayer, inputProjectionLayer, activationLayer, bias, biasUpdateRule)

}
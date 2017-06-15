package shape.komputation.layers.recurrent

import shape.komputation.initialization.InitializationStrategy
import shape.komputation.initialization.initializeRow
import shape.komputation.initialization.initializeRowVector
import shape.komputation.layers.OptimizableLayer
import shape.komputation.layers.RecurrentLayer
import shape.komputation.layers.feedforward.SharedProjectionLayer
import shape.komputation.layers.feedforward.activation.ActivationLayer
import shape.komputation.layers.feedforward.createSharedProjectionLayer
import shape.komputation.matrix.RealMatrix
import shape.komputation.matrix.createRealVector
import shape.komputation.optimization.OptimizationStrategy
import shape.komputation.optimization.UpdateRule
import shape.komputation.optimization.updateDensely

class VanillaRecurrentLayer(
    val name : String?,
    private val hiddenDimension : Int,
    private val stateProjectionLayer : SharedProjectionLayer,
    private val inputProjectionLayer : SharedProjectionLayer,
    private val activationLayer: ActivationLayer,
    private val sharedBias : DoubleArray? = null,
    private val biasUpdateRule: UpdateRule? = null) : RecurrentLayer(name), OptimizableLayer {

    var step : Int = -1

    private val state : RealMatrix = createRealVector(hiddenDimension)

    private val optimizeBias = sharedBias != null && biasUpdateRule != null

    override fun resetForward() {

        step = -1
        state.zero()

        stateProjectionLayer.resetForward()
        inputProjectionLayer.resetForward()

    }

    private var differentiationWrtBias = if(optimizeBias) DoubleArray(sharedBias!!.size) else null

    override fun resetBackward() {

        if (optimizeBias) {

            differentiationWrtBias = DoubleArray(sharedBias!!.size)
        }

        stateProjectionLayer.resetBackward()
        inputProjectionLayer.resetBackward()

    }

    // activation = activate(state weights * state + input weights * input)
    override fun forward(input : RealMatrix) : RealMatrix {

        step++

        // projected state = state weights * state
        val projectedState = stateProjectionLayer.forward(state)

        // projected input = input weights * input
        val projectedInput = inputProjectionLayer.forward(input)

        // pre-activation = projected state + projected input
        val projectedStateEntries = projectedState.getEntries()
        val projectedInputEntries = projectedInput.getEntries()

        val additionEntries = DoubleArray(hiddenDimension) { index ->

            projectedInputEntries[index] + projectedStateEntries[index]

        }

        val preActivationEntries =

            if(this.sharedBias == null) {

                additionEntries

            }
            else {


                val preActivationEntries = DoubleArray(hiddenDimension) { index ->

                    additionEntries[index] + sharedBias[index]

                }

                preActivationEntries

            }

        val preActivation = createRealVector(hiddenDimension, preActivationEntries)

        // activation = activate(pre-activation)
        val activation = activationLayer.forward(preActivation)

        return activation

    }

    // Differentiate w.r.t input
    override fun backward(chain: RealMatrix) : Pair<RealMatrix, RealMatrix> {

        // d activate(state weights * state(1) + input weights * input(2) + bias)) / d state weights * state(1) + input weights * input(2) + bias
        val backwardActivation = activationLayer.backward(chain)

        // d state weights * state(1) + input weights * input(2) + bias / d state(1) = state weights
        val differentiationWrtState = stateProjectionLayer.backward(backwardActivation)

        // d state weights * state(1) + input weights * input(2) + bias / d input(2) = input weights
        val differentiationWrtInput = inputProjectionLayer.backward(backwardActivation)

        if (optimizeBias) {

            // d state weights * state(1) + input weights * input(2) + bias / d bias = 1
            val sharedBias = this.sharedBias!!
            val differentiationWrtBias = this.differentiationWrtBias!!

            val backwardActivationEntries = backwardActivation.getEntries()

            for (index in 0..sharedBias.size - 1) {

                differentiationWrtBias[index] += backwardActivationEntries[index]

            }

        }

        step--

        return differentiationWrtState to differentiationWrtInput

    }

    override fun optimize() {

        stateProjectionLayer.optimize()
        inputProjectionLayer.optimize()

        if (optimizeBias) {

            updateDensely(this.sharedBias!!, this.differentiationWrtBias!!, biasUpdateRule!!)

        }

    }

}

fun createVanillaRecurrentLayer(
    maximumSteps : Int,
    inputDimension : Int,
    hiddenDimension: Int,
    activationLayer: ActivationLayer,
    initializationStrategy : InitializationStrategy,
    optimizationStrategy : OptimizationStrategy? = null) =

    createVanillaRecurrentLayer(null, maximumSteps, inputDimension, hiddenDimension, activationLayer, initializationStrategy, optimizationStrategy)

fun createVanillaRecurrentLayer(
    name : String?,
    maximumSteps : Int,
    inputDimension : Int,
    hiddenDimension: Int,
    activationLayer: ActivationLayer,
    initializationStrategy : InitializationStrategy,
    optimizationStrategy : OptimizationStrategy? = null): VanillaRecurrentLayer {

    val stateProjectionName = if(name == null) null else "$name-state-projection"
    val stateProjectionLayer = createSharedProjectionLayer(stateProjectionName, maximumSteps, hiddenDimension, hiddenDimension, initializationStrategy, optimizationStrategy)

    val inputProjectionName = if(name == null) null else "$name-input-projection"
    val inputProjectionLayer = createSharedProjectionLayer(inputProjectionName, maximumSteps, inputDimension, hiddenDimension, initializationStrategy, optimizationStrategy)

    val bias = initializeRow(initializationStrategy, hiddenDimension)

    val biasUpdateRule =

        if (optimizationStrategy != null) {

            optimizationStrategy(bias.size, 1)

        }
        else {

            null

        }

    return VanillaRecurrentLayer(
        name,
        hiddenDimension,
        stateProjectionLayer,
        inputProjectionLayer,
        activationLayer,
        bias,
        biasUpdateRule
    )

}
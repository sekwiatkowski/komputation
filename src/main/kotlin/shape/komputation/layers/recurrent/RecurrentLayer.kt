package shape.komputation.layers.recurrent

import shape.komputation.functions.backwardProjectionWrtBias
import shape.komputation.initialization.InitializationStrategy
import shape.komputation.initialization.initializeRowVector
import shape.komputation.layers.OptimizableLayer
import shape.komputation.layers.RecurrentLayer
import shape.komputation.layers.StatefulLayer
import shape.komputation.layers.feedforward.projection.SharedProjectionLayer
import shape.komputation.layers.feedforward.activation.ActivationLayer
import shape.komputation.layers.feedforward.projection.createSharedProjectionLayer
import shape.komputation.matrix.DoubleMatrix
import shape.komputation.matrix.doubleZeroRowVector
import shape.komputation.optimization.*
import java.util.*

class RecurrentLayer(
    val name : String?,
    private val hiddenDimension : Int,
    private val stateProjectionLayer : SharedProjectionLayer,
    private val inputProjectionLayer : SharedProjectionLayer,
    private val activationLayer: ActivationLayer,
    private val sharedBias : DoubleArray? = null,
    private val biasUpdateRule: UpdateRule? = null) : RecurrentLayer(name), OptimizableLayer, StatefulLayer {

    private var state = doubleZeroRowVector(hiddenDimension)

    private val optimizeBias = sharedBias != null && biasUpdateRule != null

    private val seriesAccumulator = if(optimizeBias) DenseAccumulator(sharedBias!!.size) else null
    private val batchAccumulator = if(optimizeBias) DenseAccumulator(sharedBias!!.size) else null

    private var isAtFirstStep = true

    override fun startForward() {

        isAtFirstStep = true

        Arrays.fill(state.entries, 0.0)

        stateProjectionLayer.startForward()
        inputProjectionLayer.startForward()

    }

    // activation = activate(state weights * state + input weights * input)
    override fun forward(input : DoubleMatrix) : DoubleMatrix {

        // projected input = input weights * input
        val projectedInput = inputProjectionLayer.forward(input)
        val projectedInputEntries = projectedInput.entries

        val additionEntries =

            if (isAtFirstStep) {

                projectedInputEntries

            }
            else {

                // projected state = state weights * state
                val projectedState = stateProjectionLayer.forward(this.state)
                val projectedStateEntries = projectedState.entries

                // addition = projected state + projected input
                DoubleArray(hiddenDimension) { index ->

                    projectedInputEntries[index] + projectedStateEntries[index]

                }

            }

        // pre-activation = addition + bias
        val preActivation =

            if(this.sharedBias == null) {

                additionEntries

            }
            else {

                DoubleArray(hiddenDimension) { index ->

                    additionEntries[index] + sharedBias[index]

                }

            }

        // activation = activate(pre-activation)
        val activation = activationLayer.forward(DoubleMatrix(hiddenDimension, 1, preActivation))

        isAtFirstStep = false

        this.state = activation

        return activation

    }

    // Differentiate w.r.t input
    override fun backward(chain: DoubleMatrix) : Pair<DoubleMatrix, DoubleMatrix> {

        // d activate(state weights * state(1) + input weights * input(2) + bias)) / d state weights * state(1) + input weights * input(2) + bias
        val backwardActivation = activationLayer.backward(chain)

        // d state weights * state(1) + input weights * input(2) + bias / d state(1) = state weights
        val backwardWrtState = stateProjectionLayer.backward(backwardActivation)

        // d state weights * state(1) + input weights * input(2) + bias / d input(2) = input weights
        val backwardWrtInput = inputProjectionLayer.backward(backwardActivation)

        if (optimizeBias) {

            // d state weights * state(1) + input weights * input(2) + bias / d bias = 1
            val backwardWrtBias = backwardProjectionWrtBias(this.sharedBias!!.size, chain.entries, chain.numberRows, chain.numberColumns)

            this.seriesAccumulator!!.accumulate(backwardWrtBias)

        }

        return backwardWrtState to backwardWrtInput

    }

    override fun finishBackward() {

        stateProjectionLayer.finishBackward()
        inputProjectionLayer.finishBackward()

        if (optimizeBias) {

            val seriesAccumulator = this.seriesAccumulator!!

            batchAccumulator!!.accumulate(seriesAccumulator.getAccumulation())

            seriesAccumulator.reset()

        }

    }

    override fun optimize() {

        stateProjectionLayer.optimize()
        inputProjectionLayer.optimize()

        if (optimizeBias) {

            val batchAccumulator = this.batchAccumulator!!

            updateDensely(this.sharedBias!!, batchAccumulator.getAccumulation(), batchAccumulator.getCount(), biasUpdateRule!!)

            batchAccumulator.reset()

        }

    }

}

fun createRecurrentLayer(
    inputDimension : Int,
    hiddenDimension: Int,
    activationLayer: ActivationLayer,
    stateWeightInitializationStrategy: InitializationStrategy,
    inputWeightInitializationStrategy: InitializationStrategy,
    biasInitializationStrategy: InitializationStrategy,
    optimizationStrategy : OptimizationStrategy? = null) =

    createRecurrentLayer(null, inputDimension, hiddenDimension, activationLayer, stateWeightInitializationStrategy, inputWeightInitializationStrategy, biasInitializationStrategy, optimizationStrategy)

fun createRecurrentLayer(
    name : String?,
    inputDimension : Int,
    hiddenDimension: Int,
    activationLayer: ActivationLayer,
    stateWeightInitializationStrategy: InitializationStrategy,
    inputWeightInitializationStrategy: InitializationStrategy,
    biasInitializationStrategy : InitializationStrategy,
    optimizationStrategy : OptimizationStrategy? = null): RecurrentLayer {

    val stateProjectionName = if(name == null) null else "$name-state-projection"
    val stateProjectionLayer = createSharedProjectionLayer(stateProjectionName, hiddenDimension, hiddenDimension, stateWeightInitializationStrategy, optimizationStrategy)

    val inputProjectionName = if(name == null) null else "$name-input-projection"
    val inputProjectionLayer = createSharedProjectionLayer(inputProjectionName, inputDimension, hiddenDimension, inputWeightInitializationStrategy, optimizationStrategy)

    val bias = initializeRowVector(biasInitializationStrategy, hiddenDimension)

    val biasUpdateRule =

        if (optimizationStrategy != null) {

            optimizationStrategy(bias.size, 1)

        }
        else {

            null

        }

    return RecurrentLayer(
        name,
        hiddenDimension,
        stateProjectionLayer,
        inputProjectionLayer,
        activationLayer,
        bias,
        biasUpdateRule
    )

}
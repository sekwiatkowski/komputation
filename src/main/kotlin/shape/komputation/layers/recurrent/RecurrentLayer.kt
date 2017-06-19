package shape.komputation.layers.recurrent

import shape.komputation.functions.activation.ActivationFunction
import shape.komputation.initialization.InitializationStrategy
import shape.komputation.layers.FeedForwardLayer
import shape.komputation.layers.OptimizableLayer
import shape.komputation.layers.feedforward.activation.ActivationLayer
import shape.komputation.layers.feedforward.activation.ReluLayer
import shape.komputation.layers.feedforward.activation.SigmoidLayer
import shape.komputation.layers.feedforward.activation.SoftmaxLayer
import shape.komputation.matrix.*
import shape.komputation.optimization.OptimizationStrategy
import java.util.*

class RecurrentLayer(
    name : String?,
    private val hiddenDimension : Int,
    private val activationLayers: Array<ActivationLayer>,
    private val stateProjection: SeriesProjection,
    private val inputProjection: SeriesProjection,
    private val bias : SeriesBias?) : FeedForwardLayer(name), OptimizableLayer {

    private var state = doubleZeroRowVector(hiddenDimension)
    private var input : SequenceMatrix = EMPTY_SEQUENCE_MATRIX

    override fun forward(input: DoubleMatrix): DoubleMatrix {

        Arrays.fill(state.entries, 0.0)

        input as SequenceMatrix

        this.input = input

        var output : DoubleMatrix? = null

        for (indexStep in 0..input.numberSteps - 1) {

            val step = input.getStep(indexStep)

            // projected input = input weights * input
            val projectedInput = inputProjection.forwardStep(indexStep, step)
            val projectedInputEntries = projectedInput.entries

            // projected state = state weights * state
            val projectedState =  stateProjection.forwardStep(indexStep, this.state)
            val projectedStateEntries = projectedState.entries

            // addition = projected state + projected input
            val additionEntries = DoubleArray(hiddenDimension) { index ->

                projectedInputEntries[index] + projectedStateEntries[index]

            }

            // pre-activation = addition + bias
            val preActivation =

                if(this.bias == null) {

                    additionEntries

                }
                else {

                    this.bias.forwardStep(additionEntries)

                }

            // activation = activate(pre-activation)
            output = activationLayers[indexStep].forward(DoubleMatrix(hiddenDimension, 1, preActivation))

            this.state = output


        }

        return output!!

    }

    override fun backward(chain: DoubleMatrix): DoubleMatrix {

        var seriesChain = chain

        val seriesBackwardWrtInput = zeroSequenceMatrix(this.input.numberSteps, this.input.numberStepRows, this.input.numberStepColumns)

        for (indexStep in this.input.numberSteps - 1 downTo 0) {

            // d activate(state weights * state(1) + input weights * input(2) + bias)) / d state weights * state(1) + input weights * input(2) + bias
            val backwardActivation = this.activationLayers[indexStep].backward(seriesChain)

            // d state weights * state(1) + input weights * input(2) + bias / d state(1) = state weights
            seriesChain = this.stateProjection.backwardStep(indexStep, backwardActivation)

            // d state weights * state(1) + input weights * input(2) + bias / d input(2) = input weights
            val backwardWrtInput = this.inputProjection.backwardStep(indexStep, backwardActivation)

            if (this.bias != null) {
                this.bias.backwardStep(seriesChain)
            }

            seriesBackwardWrtInput.setStep(indexStep, backwardWrtInput.entries)

        }

        this.stateProjection.backwardSeries()
        this.inputProjection.backwardSeries()

        if (this.bias != null) {

            this.bias.backwardSeries()

        }


        return seriesChain

    }

    override fun optimize() {

        this.stateProjection.optimize()
        this.inputProjection.optimize()

        if (this.bias != null) {

            this.bias.optimize()

        }

    }

}

fun createRecurrentLayer(
    numberSteps : Int,
    stepSize : Int,
    hiddenDimension: Int,
    activationFunction : ActivationFunction,
    stateWeightInitializationStrategy: InitializationStrategy,
    inputWeightInitializationStrategy: InitializationStrategy,
    biasInitializationStrategy: InitializationStrategy,
    optimizationStrategy : OptimizationStrategy? = null) =

    createRecurrentLayer(
        null,
        numberSteps,
        stepSize,
        hiddenDimension,
        activationFunction,
        stateWeightInitializationStrategy,
        inputWeightInitializationStrategy,
        biasInitializationStrategy,
        optimizationStrategy)

fun createRecurrentLayer(
    name : String?,
    numberSteps : Int,
    stepSize : Int,
    hiddenDimension: Int,
    activationFunction : ActivationFunction,
    stateWeightInitializationStrategy: InitializationStrategy,
    inputWeightInitializationStrategy: InitializationStrategy,
    biasInitializationStrategy: InitializationStrategy,
    optimizationStrategy : OptimizationStrategy? = null): RecurrentLayer {

    val activationLayers = Array(numberSteps) { index ->

        val activationLayerName = if(name == null) null else "$name-activation-$index"

        when (activationFunction) {

            ActivationFunction.Sigmoid -> SigmoidLayer(activationLayerName)
            ActivationFunction.ReLU -> ReluLayer(activationLayerName)
            ActivationFunction.Softmax -> SoftmaxLayer(activationLayerName)

        }

    }

    val stateProjectionName = if(name == null) null else "$name-state-projection"
    val seriesProjection = createSeriesProjection(stateProjectionName, numberSteps, true, hiddenDimension, hiddenDimension, stateWeightInitializationStrategy, optimizationStrategy)

    val inputProjectionName = if(name == null) null else "$name-input-projection"
    val inputProjection = createSeriesProjection(inputProjectionName, numberSteps, false, hiddenDimension, stepSize, inputWeightInitializationStrategy, optimizationStrategy)

    val biasName = if(name == null) null else "$name-input-projection"
    val bias = createSeriesBias(biasName, hiddenDimension, biasInitializationStrategy, optimizationStrategy)

    return RecurrentLayer(
        name,
        hiddenDimension,
        activationLayers,
        seriesProjection,
        inputProjection,
        bias
    )

}
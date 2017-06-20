package shape.komputation.layers.recurrent

import shape.komputation.functions.activation.ActivationFunction
import shape.komputation.initialization.InitializationStrategy
import shape.komputation.layers.FeedForwardLayer
import shape.komputation.layers.OptimizableLayer
import shape.komputation.layers.feedforward.activation.*
import shape.komputation.matrix.*
import shape.komputation.optimization.OptimizationStrategy
import java.util.*

class Encoder(
    name : String?,
    private val numberSteps: Int,
    private val numberStepRows: Int,
    private val numberStepColumns: Int,
    private val hiddenDimension : Int,
    private val inputProjection: SeriesProjection,
    private val previousStateProjection: SeriesProjection,
    private val activationLayers: Array<ActivationLayer>,
    private val bias : SeriesBias?) : FeedForwardLayer(name), OptimizableLayer {

    private var state = doubleZeroColumnVector(hiddenDimension)

    override fun forward(input: DoubleMatrix): DoubleMatrix {

        Arrays.fill(state.entries, 0.0)

        input as SequenceMatrix

        var output : DoubleMatrix? = null

        for (indexStep in 0..numberSteps - 1) {

            val step = input.getStep(indexStep)

            // projected input = input weights * input
            val projectedInput = inputProjection.forwardStep(indexStep, step)
            val projectedInputEntries = projectedInput.entries

            // projected state = state weights * state
            val projectedState =  previousStateProjection.forwardStep(indexStep, this.state)
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

        val seriesBackwardWrtInput = zeroSequenceMatrix(this.numberSteps, this.numberStepRows, this.numberStepColumns)

        for (indexStep in this.numberSteps - 1 downTo 0) {

            // d activate(state weights * state(1) + input weights * input(2) + bias)) / d state weights * state(1) + input weights * input(2) + bias
            val backwardActivation = this.activationLayers[indexStep].backward(seriesChain)

            // d state weights * state(1) + input weights * input(2) + bias / d state(1) = state weights
            seriesChain = this.previousStateProjection.backwardStep(indexStep, backwardActivation)

            // d state weights * state(1) + input weights * input(2) + bias / d input(2) = input weights
            val backwardWrtInput = this.inputProjection.backwardStep(indexStep, backwardActivation)

            if (this.bias != null) {

                // d state weights * state(1) + input weights * input(2) + bias / d bias = 1
                this.bias.backwardStep(backwardActivation)

            }

            seriesBackwardWrtInput.setStep(indexStep, backwardWrtInput.entries)

        }

        this.previousStateProjection.backwardSeries()
        this.inputProjection.backwardSeries()

        if (this.bias != null) {

            this.bias.backwardSeries()

        }

        return seriesChain

    }

    override fun optimize() {

        this.previousStateProjection.optimize()
        this.inputProjection.optimize()

        if (this.bias != null) {

            this.bias.optimize()

        }

    }

}

fun createEncoder(
    numberSteps : Int,
    numberStepRows : Int,
    hiddenDimension: Int,
    activationFunction : ActivationFunction,
    inputWeightInitializationStrategy: InitializationStrategy,
    stateWeightInitializationStrategy: InitializationStrategy,
    biasInitializationStrategy: InitializationStrategy?,
    optimizationStrategy : OptimizationStrategy? = null) =

    createEncoder(
        null,
        numberSteps,
        numberStepRows,
        hiddenDimension,
        activationFunction,
        inputWeightInitializationStrategy,
        stateWeightInitializationStrategy,
        biasInitializationStrategy,
        optimizationStrategy)

fun createEncoder(
    name : String?,
    numberSteps : Int,
    numberStepRows : Int,
    hiddenDimension: Int,
    activationFunction : ActivationFunction,
    inputProjectionInitializationStrategy: InitializationStrategy,
    previousStateProjectionInitializationStrategy: InitializationStrategy,
    biasInitializationStrategy: InitializationStrategy?,
    optimizationStrategy : OptimizationStrategy? = null): Encoder {

    val inputProjectionName = if(name == null) null else "$name-input-projection"
    val inputProjection = createSeriesProjection(inputProjectionName, numberSteps, false, hiddenDimension, numberStepRows, inputProjectionInitializationStrategy, optimizationStrategy)

    val previousStateProjectionName = if(name == null) null else "$name-previous-state-projection"
    val previousStateProjection = createSeriesProjection(previousStateProjectionName, numberSteps, true, hiddenDimension, hiddenDimension, previousStateProjectionInitializationStrategy, optimizationStrategy)

    val activationName = if(name == null) null else "$name-state-activation"
    val activationLayers = createActivationLayers(activationName, numberSteps, activationFunction)

    val bias =

        if(biasInitializationStrategy == null)
            null
        else
            createSeriesBias(if(name == null) null else "$name-bias", hiddenDimension, biasInitializationStrategy, optimizationStrategy)

    return Encoder(
        name,
        numberSteps,
        numberStepRows,
        1,
        hiddenDimension,
        inputProjection,
        previousStateProjection,
        activationLayers,
        bias
    )

}

package shape.komputation.layers.feedforward.encoder

import shape.komputation.functions.activation.ActivationFunction
import shape.komputation.initialization.InitializationStrategy
import shape.komputation.layers.ContinuationLayer
import shape.komputation.layers.OptimizableLayer
import shape.komputation.layers.combination.AdditionCombination
import shape.komputation.layers.concatenateNames
import shape.komputation.layers.feedforward.activation.createActivationLayers
import shape.komputation.layers.feedforward.recurrent.SeriesBias
import shape.komputation.layers.feedforward.recurrent.SeriesProjection
import shape.komputation.layers.feedforward.recurrent.createSeriesBias
import shape.komputation.layers.feedforward.recurrent.createSeriesProjection
import shape.komputation.matrix.*
import shape.komputation.optimization.OptimizationStrategy

class SingleOutputEncoder(
    name : String?,
    private val steps : Array<RecurrentUnit>,
    private val numberSteps: Int,
    private val inputDimension: Int,
    private val hiddenDimension : Int,
    private val inputProjection: SeriesProjection,
    private val previousStateProjection: SeriesProjection,
    private val bias : SeriesBias?) : ContinuationLayer(name), OptimizableLayer {

    override fun forward(input: DoubleMatrix): DoubleMatrix {

        var state = doubleZeroColumnVector(hiddenDimension)

        input as SequenceMatrix

        var output = EMPTY_DOUBLE_MATRIX

        for (indexStep in 0..numberSteps - 1) {

            val stepInput = input.getStep(indexStep)

            state = this.steps[indexStep].forward(state, stepInput)

            if (indexStep == numberSteps - 1) {

                output =  state

            }

        }

        return output

    }

    override fun backward(incoming: DoubleMatrix): DoubleMatrix {

        val seriesBackwardWrtInput = zeroSequenceMatrix(this.numberSteps, this.inputDimension)

        var stateChain : DoubleArray? = null

        val incomingEntries = incoming.entries

        for (indexStep in this.numberSteps - 1 downTo 0) {

            val backwardOutput =

                if (indexStep + 1 == this.numberSteps) {

                    incomingEntries

                }
                else {

                    null

                }

            val (backwardStatePreActivationWrtPreviousState, backwardStatePreActivationWrtInput) = this.steps[indexStep].backward(stateChain, backwardOutput)

            stateChain = backwardStatePreActivationWrtPreviousState

            seriesBackwardWrtInput.setStep(indexStep, backwardStatePreActivationWrtInput)

        }

        this.previousStateProjection.backwardSeries()
        this.inputProjection.backwardSeries()

        this.bias?.backwardSeries()

        return seriesBackwardWrtInput

    }

    override fun optimize() {

        this.previousStateProjection.optimize()
        this.inputProjection.optimize()

        this.bias?.optimize()

    }

}

fun createSingleOutputEncoder(
    numberSteps : Int,
    numberStepRows : Int,
    hiddenDimension: Int,
    inputWeightInitializationStrategy: InitializationStrategy,
    stateWeightInitializationStrategy: InitializationStrategy,
    biasInitializationStrategy: InitializationStrategy?,
    activationFunction : ActivationFunction,
    optimizationStrategy : OptimizationStrategy? = null) =

    createSingleOutputEncoder(
        null,
        numberSteps,
        numberStepRows,
        hiddenDimension,
        inputWeightInitializationStrategy,
        stateWeightInitializationStrategy,
        biasInitializationStrategy,
        activationFunction,
        optimizationStrategy)

fun createSingleOutputEncoder(
    name : String?,
    numberSteps : Int,
    inputDimension : Int,
    hiddenDimension: Int,
    inputProjectionInitializationStrategy: InitializationStrategy,
    previousStateProjectionInitializationStrategy: InitializationStrategy,
    biasInitializationStrategy: InitializationStrategy?,
    activationFunction : ActivationFunction,
    optimizationStrategy : OptimizationStrategy? = null): SingleOutputEncoder {

    val inputProjectionSeriesName = concatenateNames(name, "input-projection")
    val inputProjectionStepName = concatenateNames(name, "input-projection-step")

    val (inputProjectionSeries, inputProjectionSteps) = createSeriesProjection(
        inputProjectionSeriesName,
        inputProjectionStepName,
        numberSteps,
        false,
        inputDimension,
        hiddenDimension,
        inputProjectionInitializationStrategy,
        optimizationStrategy)

    val previousStateProjectionSeriesName = concatenateNames(name, "previous-state-projection")
    val previousStateProjectionStepName = concatenateNames(name, "previous-state-projection-step")

    val (previousStateProjectionSeries, previousStateProjectionSteps) = createSeriesProjection(
        previousStateProjectionSeriesName,
        previousStateProjectionStepName,
        numberSteps,
        true,
        hiddenDimension, hiddenDimension,
        previousStateProjectionInitializationStrategy,
        optimizationStrategy)

    val additions = Array(numberSteps) { indexStep ->

        val additionName = concatenateNames(name, "addition-step-$indexStep")
        AdditionCombination(additionName)

    }

    val bias =

        if(biasInitializationStrategy == null)
            null
        else
            createSeriesBias(concatenateNames(name, "bias"), hiddenDimension, biasInitializationStrategy, optimizationStrategy)

    val activationName = concatenateNames(name, "state-activation")
    val activationLayers = createActivationLayers(numberSteps, activationName, activationFunction)

    val encoderSteps = Array(numberSteps) { indexStep ->

        val encoderStepName = concatenateNames(name, "step-$indexStep")

        RecurrentUnit(encoderStepName, inputProjectionSteps[indexStep], previousStateProjectionSteps[indexStep], additions[indexStep], bias, activationLayers[indexStep])

    }

    val encoder = SingleOutputEncoder(
        name,
        encoderSteps,
        numberSteps,
        inputDimension,
        hiddenDimension,
        inputProjectionSeries,
        previousStateProjectionSeries,
        bias
    )

    return encoder

}

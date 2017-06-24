package shape.komputation.layers.feedforward.encoder

import shape.komputation.functions.activation.ActivationFunction
import shape.komputation.initialization.InitializationStrategy
import shape.komputation.layers.ContinuationLayer
import shape.komputation.layers.OptimizableLayer
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
    private val steps : Array<EncoderStep>,
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

        var stateChain : DoubleMatrix? = null

        for (indexStep in this.numberSteps - 1 downTo 0) {

            val backwardOutput =

                if (indexStep + 1 == this.numberSteps) {

                    incoming

                }
                else {

                    null

                }

            val (backwardStatePreActivationWrtPreviousState, backwardStatePreActivationWrtInput) = this.steps[indexStep].backward(stateChain, backwardOutput)

            stateChain = backwardStatePreActivationWrtPreviousState

            seriesBackwardWrtInput.setStep(indexStep, backwardStatePreActivationWrtInput.entries)

        }

        this.previousStateProjection.backwardSeries()
        this.inputProjection.backwardSeries()

        this.bias?.backwardSeries()

        return stateChain!!

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

    val inputProjectionName = concatenateNames(name, "input-projection")
    val (inputProjectionSeries, inputProjectionSteps) = createSeriesProjection(
        inputProjectionName,
        numberSteps,
        false,
        inputDimension,
        hiddenDimension,
        inputProjectionInitializationStrategy,
        optimizationStrategy)

    val previousStateProjectionName = concatenateNames(name, "previous-state-projection")
    val (previousStateProjectionSeries, previousStateProjectionSteps) = createSeriesProjection(
        previousStateProjectionName,
        numberSteps,
        true,
        hiddenDimension, hiddenDimension,
        previousStateProjectionInitializationStrategy,
        optimizationStrategy)

    val activationName = concatenateNames(name, "state-activation")
    val activationLayers = createActivationLayers(numberSteps, activationName, activationFunction)

    val bias =

        if(biasInitializationStrategy == null)
            null
        else
            createSeriesBias(concatenateNames(name, "bias"), hiddenDimension, biasInitializationStrategy, optimizationStrategy)

    val encoderSteps = Array(numberSteps) { indexStep ->

        val encoderStepName = concatenateNames(name, "step-$indexStep")

        EncoderStep(encoderStepName, hiddenDimension, inputProjectionSteps[indexStep], previousStateProjectionSteps[indexStep], bias, activationLayers[indexStep])

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

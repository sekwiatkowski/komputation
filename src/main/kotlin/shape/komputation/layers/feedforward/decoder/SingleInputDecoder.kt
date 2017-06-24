package shape.komputation.layers.feedforward.decoder

import shape.komputation.functions.activation.ActivationFunction
import shape.komputation.functions.extractStep
import shape.komputation.initialization.InitializationStrategy
import shape.komputation.layers.ContinuationLayer
import shape.komputation.layers.OptimizableLayer
import shape.komputation.layers.concatenateNames
import shape.komputation.layers.feedforward.activation.createActivationLayer
import shape.komputation.layers.feedforward.recurrent.SeriesBias
import shape.komputation.layers.feedforward.recurrent.SeriesProjection
import shape.komputation.layers.feedforward.recurrent.createSeriesBias
import shape.komputation.layers.feedforward.recurrent.createSeriesProjection
import shape.komputation.matrix.DoubleMatrix
import shape.komputation.matrix.doubleZeroColumnVector
import shape.komputation.matrix.zeroSequenceMatrix
import shape.komputation.optimization.OptimizationStrategy

class SingleInputDecoder(
    name : String?,
    private val numberSteps : Int,
    private val decoderSteps : Array<DecoderStep>,
    private val inputDimension: Int,
    private val outputDimension : Int,
    private val previousOutputProjection: SeriesProjection,
    private val stateProjection: SeriesProjection,
    private val outputProjection: SeriesProjection,
    private val bias : SeriesBias?) : ContinuationLayer(name), OptimizableLayer {

    override fun forward(input: DoubleMatrix): DoubleMatrix {

        val seriesOutput = zeroSequenceMatrix(this.numberSteps, this.inputDimension)

        var state = input
        var previousOutput = doubleZeroColumnVector(outputDimension)

        for (indexStep in 0..this.numberSteps - 1) {

            val decoderStep = this.decoderSteps[indexStep]

            val (newState, newOutput) = decoderStep.forward(state, previousOutput)

            seriesOutput.setStep(indexStep, newOutput.entries)

            state = newState
            previousOutput = newOutput

        }

        return seriesOutput

    }

    // Incoming gradient: d chain / d series prediction
    override fun backward(chain: DoubleMatrix): DoubleMatrix {

        val chainEntries = chain.entries

        var backwardStatePreActivationWrtPreviousOutput : DoubleMatrix? = null
        var backwardStatePreActivationWrtPreviousState : DoubleMatrix? = null

        for (indexStep in this.numberSteps - 1 downTo 0) {

            val stepChain = extractStep(chainEntries, indexStep, outputDimension)

            val decoderStep = this.decoderSteps[indexStep]

            val (newBackwardStatePreActivationWrtPreviousOutput, newBackwardStatePreActivationWrtPreviousState) = decoderStep.backward(stepChain, backwardStatePreActivationWrtPreviousOutput, backwardStatePreActivationWrtPreviousState)

            backwardStatePreActivationWrtPreviousOutput = newBackwardStatePreActivationWrtPreviousOutput
            backwardStatePreActivationWrtPreviousState = newBackwardStatePreActivationWrtPreviousState

        }

        this.outputProjection.backwardSeries()
        this.stateProjection.backwardSeries()
        this.previousOutputProjection.backwardSeries()

        this.bias?.backwardSeries()

        return backwardStatePreActivationWrtPreviousState!!

    }

    override fun optimize() {

        this.outputProjection.optimize()
        this.stateProjection.optimize()
        this.previousOutputProjection.optimize()

        this.bias?.optimize()

    }

}

fun createSingleInputDecoder(
    numberSteps: Int,
    inputDimension: Int,
    hiddenDimension: Int,
    outputDimension: Int,
    previousOutputProjectionInitializationStrategy: InitializationStrategy,
    previousStateProjectionInitializationStrategy: InitializationStrategy,
    biasInitializationStrategy: InitializationStrategy?,
    stateActivation: ActivationFunction,
    outputProjectionInitializationStrategy: InitializationStrategy,
    outputActivation: ActivationFunction,
    optimizationStrategy: OptimizationStrategy?) =

    createSingleInputDecoder(
        null,
        numberSteps,
        inputDimension,
        hiddenDimension,
        outputDimension,
        previousOutputProjectionInitializationStrategy,
        previousStateProjectionInitializationStrategy,
        biasInitializationStrategy,
        stateActivation,
        outputProjectionInitializationStrategy,
        outputActivation,
        optimizationStrategy)


fun createSingleInputDecoder(
    name : String?,
    numberSteps: Int,
    inputDimension: Int,
    hiddenDimension: Int,
    outputDimension: Int,
    previousOutputProjectionInitializationStrategy: InitializationStrategy,
    previousStateProjectionInitializationStrategy: InitializationStrategy,
    biasInitializationStrategy: InitializationStrategy?,
    stateActivationFunction: ActivationFunction,
    outputProjectionInitializationStrategy: InitializationStrategy,
    outputActivationFunction: ActivationFunction,
    optimizationStrategy: OptimizationStrategy?): SingleInputDecoder {

    val previousOutputProjectionName = concatenateNames(name, "previous-output-projection")
    val (previousOutputProjectionSeries, previousOutputProjectionSteps) = createSeriesProjection(previousOutputProjectionName, numberSteps, true, outputDimension, hiddenDimension, previousOutputProjectionInitializationStrategy, optimizationStrategy)

    val previousStateProjectionName = concatenateNames(name, "previous-state-projection")
    val (previousStateProjectionSeries, previousStateProjectionSteps) = createSeriesProjection(previousStateProjectionName, numberSteps, false, hiddenDimension, hiddenDimension, previousStateProjectionInitializationStrategy, optimizationStrategy)

    val outputProjectionName = concatenateNames(name, "output-projection")
    val (outputProjectionSeries, outputProjectionSteps) = createSeriesProjection(outputProjectionName, numberSteps, false, hiddenDimension, outputDimension, outputProjectionInitializationStrategy, optimizationStrategy)

    val bias =

        if(biasInitializationStrategy == null)
            null
        else {

            val biasName = concatenateNames(name, "bias")

            createSeriesBias(biasName, hiddenDimension, biasInitializationStrategy, optimizationStrategy)

        }


    val decoderSteps = Array(numberSteps) { indexStep ->

        val stepName = concatenateNames(name, "step-$indexStep")
        val isLastStep = indexStep + 1 == numberSteps

        val previousOutputProjectionStep = previousOutputProjectionSteps[indexStep]

        val previousStateProjectionStep = previousStateProjectionSteps[indexStep]

        val stateActivationName = concatenateNames(stepName, "state-activation")
        val stateActivation = createActivationLayer(stateActivationName, stateActivationFunction)

        val outputProjection = outputProjectionSteps[indexStep]

        val outputActivationName = concatenateNames(stepName, "output-activation")
        val outputActivation = createActivationLayer(outputActivationName, outputActivationFunction)

        DecoderStep(
            stepName,
            isLastStep,
            hiddenDimension,
            outputDimension,
            previousOutputProjectionStep,
            previousStateProjectionStep,
            stateActivation,
            outputProjection,
            outputActivation,
            bias
        )

    }


    val decoder = SingleInputDecoder(
        name,
        numberSteps,
        decoderSteps,
        inputDimension,
        outputDimension,
        previousOutputProjectionSeries,
        previousStateProjectionSeries,
        outputProjectionSeries,
        bias)

    return decoder

}
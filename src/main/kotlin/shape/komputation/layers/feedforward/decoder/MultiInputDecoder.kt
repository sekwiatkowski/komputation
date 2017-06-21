package shape.komputation.layers.feedforward.decoder

import shape.komputation.functions.activation.ActivationFunction
import shape.komputation.functions.extractStep
import shape.komputation.initialization.InitializationStrategy
import shape.komputation.layers.ContinuationLayer
import shape.komputation.layers.OptimizableLayer
import shape.komputation.layers.feedforward.activation.createActivationLayer
import shape.komputation.layers.feedforward.recurrent.SeriesBias
import shape.komputation.layers.feedforward.recurrent.SeriesProjection
import shape.komputation.layers.feedforward.recurrent.createSeriesBias
import shape.komputation.layers.feedforward.recurrent.createSeriesProjection
import shape.komputation.matrix.DoubleMatrix
import shape.komputation.matrix.SequenceMatrix
import shape.komputation.matrix.doubleZeroColumnVector
import shape.komputation.matrix.zeroSequenceMatrix
import shape.komputation.optimization.OptimizationStrategy

class MultiInputDecoder(
    name : String?,
    private val numberSteps : Int,
    private val decoderSteps : Array<DecoderStep>,
    private val inputDimension: Int,
    private val hiddenDimension: Int,
    private val outputDimension : Int,
    private val inputProjection: SeriesProjection,
    private val stateProjection: SeriesProjection,
    private val outputProjection: SeriesProjection,
    private val bias : SeriesBias?) : ContinuationLayer(name), OptimizableLayer {

    override fun forward(input: DoubleMatrix): DoubleMatrix {

        input as SequenceMatrix

        val seriesOutput = zeroSequenceMatrix(this.numberSteps, this.outputDimension)

        // Start with a zero state
        var state = doubleZeroColumnVector(hiddenDimension)

        for (indexStep in 0..this.numberSteps - 1) {

            // Extract the n-th step input
            val stepInput = input.getStep(indexStep)

            // Access the n-th decoder step
            val decoderStep = this.decoderSteps[indexStep]

            // Forward the current state together the current step input
            val (newState, newOutput) = decoderStep.forward(state, stepInput)

            // Store the n-th output
            seriesOutput.setStep(indexStep, newOutput.entries)

            state = newState

        }

        return seriesOutput

    }

    // Incoming gradient: d chain / d series prediction
    override fun backward(chain: DoubleMatrix): DoubleMatrix {

        val chainEntries = chain.entries

        val backwardStatePreActivationWrtInput = zeroSequenceMatrix(numberSteps, inputDimension)
        var backwardStatePreActivationWrtPreviousState : DoubleMatrix? = null

        for (indexStep in this.numberSteps - 1 downTo 0) {

            val stepChain = extractStep(chainEntries, indexStep, outputDimension)

            val decoderStep = this.decoderSteps[indexStep]

            val (newBackwardStatePreActivationWrtInput, newBackwardStatePreActivationWrtPreviousState) = decoderStep.backward(stepChain, backwardStatePreActivationWrtInput, backwardStatePreActivationWrtPreviousState)

            backwardStatePreActivationWrtInput.setStep(indexStep, newBackwardStatePreActivationWrtInput.entries)
            backwardStatePreActivationWrtPreviousState = newBackwardStatePreActivationWrtPreviousState

        }

        this.outputProjection.backwardSeries()
        this.stateProjection.backwardSeries()
        this.inputProjection.backwardSeries()

        if (this.bias != null) {

            this.bias.backwardSeries()

        }

        return backwardStatePreActivationWrtInput

    }

    override fun optimize() {

        this.outputProjection.optimize()
        this.stateProjection.optimize()
        this.inputProjection.optimize()

        if (this.bias != null) {

            this.bias.optimize()

        }

    }

}

fun createMultiInputDecoder(
    numberSteps: Int,
    inputDimension: Int,
    hiddenDimension: Int,
    outputDimension: Int,
    inputWeightInitializationStrategy: InitializationStrategy,
    previousStateWeightInitializationStrategy: InitializationStrategy,
    biasInitializationStrategy: InitializationStrategy?,
    stateActivation: ActivationFunction,
    outputProjectionInitializationStrategy: InitializationStrategy,
    outputActivation: ActivationFunction,
    optimizationStrategy: OptimizationStrategy?) =

    createMultiInputDecoder(
        null,
        numberSteps,
        inputDimension,
        hiddenDimension,
        outputDimension,
        inputWeightInitializationStrategy,
        previousStateWeightInitializationStrategy,
        biasInitializationStrategy,
        stateActivation,
        outputProjectionInitializationStrategy,
        outputActivation,
        optimizationStrategy)


fun createMultiInputDecoder(
    name : String?,
    numberSteps: Int,
    inputDimension: Int,
    hiddenDimension: Int,
    outputDimension: Int,
    inputWeightInitializationStrategy: InitializationStrategy,
    previousStateWeightInitializationStrategy: InitializationStrategy,
    biasInitializationStrategy: InitializationStrategy?,
    stateActivationFunction: ActivationFunction,
    outputProjectionInitializationStrategy: InitializationStrategy,
    outputActivationFunction: ActivationFunction,
    optimizationStrategy: OptimizationStrategy?): MultiInputDecoder {

    val inputProjectionName = if(name == null) null else "$name-input-projection"
    val (inputProjectionSeries, inputProjectionSteps) = createSeriesProjection(inputProjectionName, numberSteps, false, inputDimension, hiddenDimension, inputWeightInitializationStrategy, optimizationStrategy)

    val previousStateProjectionName = if(name == null) null else "$name-previous-state-projection"
    // Don't project state at the first step
    val (previousStateProjectionSeries, previousStateProjectionSteps) = createSeriesProjection(previousStateProjectionName, numberSteps, true, hiddenDimension, hiddenDimension, previousStateWeightInitializationStrategy, optimizationStrategy)

    val outputProjectionName = if(name == null) null else "$name-output-projection"
    val (outputProjectionSeries, outputProjectionSteps) = createSeriesProjection(outputProjectionName, numberSteps, false, hiddenDimension, outputDimension, outputProjectionInitializationStrategy, optimizationStrategy)

    val bias =

        if(biasInitializationStrategy == null)
            null
        else {

            val biasName = if (name == null) null else "$name-bias"

            createSeriesBias(biasName, hiddenDimension, biasInitializationStrategy, optimizationStrategy)

        }


    val decoderSteps = Array(numberSteps) { indexStep ->

        val stepName = if(name == null) null else "$name-step-$indexStep"
        val isLastStep = indexStep + 1 == numberSteps

        val inputProjectionStep = inputProjectionSteps[indexStep]

        val previousStateProjectionStep = previousStateProjectionSteps[indexStep]

        val stateActivationName = if(name == null) null else "$name-state-activation"
        val stateActivation = createActivationLayer(stateActivationName, stateActivationFunction)

        val outputProjection = outputProjectionSteps[indexStep]

        val outputActivationName = if(name == null) null else "$name-output-activation"
        val outputActivation = createActivationLayer(outputActivationName, outputActivationFunction)

        DecoderStep(
            stepName,
            isLastStep,
            hiddenDimension,
            outputDimension,
            inputProjectionStep,
            previousStateProjectionStep,
            stateActivation,
            outputProjection,
            outputActivation,
            bias
        )

    }


    val decoder = MultiInputDecoder(
        name,
        numberSteps,
        decoderSteps,
        inputDimension,
        hiddenDimension,
        outputDimension,
        inputProjectionSeries,
        previousStateProjectionSeries,
        outputProjectionSeries,
        bias)

    return decoder

}
package shape.komputation.layers.feedforward.encoder

import shape.komputation.functions.activation.ActivationFunction
import shape.komputation.functions.extractStep
import shape.komputation.initialization.InitializationStrategy
import shape.komputation.layers.ContinuationLayer
import shape.komputation.layers.OptimizableLayer
import shape.komputation.layers.concatenateNames
import shape.komputation.layers.feedforward.activation.createActivationLayers
import shape.komputation.layers.feedforward.recurrent.SeriesBias
import shape.komputation.layers.feedforward.recurrent.SeriesProjection
import shape.komputation.layers.feedforward.recurrent.createSeriesBias
import shape.komputation.layers.feedforward.recurrent.createSeriesProjection
import shape.komputation.matrix.DoubleMatrix
import shape.komputation.matrix.SequenceMatrix
import shape.komputation.matrix.doubleZeroColumnVector
import shape.komputation.matrix.zeroSequenceMatrix
import shape.komputation.optimization.OptimizationStrategy

class MultiOutputDecoder(
    name : String?,
    private val steps : Array<EncoderStep>,
    private val numberSteps: Int,
    private val inputRows: Int,
    private val hiddenDimension : Int,
    private val inputProjection: SeriesProjection,
    private val previousStateProjection: SeriesProjection,
    private val bias : SeriesBias?) : ContinuationLayer(name), OptimizableLayer {

    override fun forward(input: DoubleMatrix): DoubleMatrix {

        var state = doubleZeroColumnVector(hiddenDimension)

        input as SequenceMatrix

        val output = zeroSequenceMatrix(numberSteps, hiddenDimension, 1)

        for (indexStep in 0..numberSteps - 1) {

            val stepInput = input.getStep(indexStep)

            state = this.steps[indexStep].forward(state, stepInput)

            output.setStep(indexStep, state.entries)


        }

        return output

    }

    override fun backward(incoming: DoubleMatrix): DoubleMatrix {

        val seriesBackwardWrtInput = zeroSequenceMatrix(this.numberSteps, inputRows)

        var stateChain : DoubleMatrix? = null

        for (indexStep in this.numberSteps - 1 downTo 0) {

            val backwardOutput = DoubleMatrix(hiddenDimension, 1, extractStep(incoming.entries, indexStep, hiddenDimension))

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

fun createMultiOutputEncoder(
    numberSteps : Int,
    numberStepRows : Int,
    hiddenDimension: Int,
    inputWeightInitializationStrategy: InitializationStrategy,
    stateWeightInitializationStrategy: InitializationStrategy,
    biasInitializationStrategy: InitializationStrategy?,
    activationFunction : ActivationFunction,
    optimizationStrategy : OptimizationStrategy? = null) =

    createMultiOutputEncoder(
        null,
        numberSteps,
        numberStepRows,
        hiddenDimension,
        inputWeightInitializationStrategy,
        stateWeightInitializationStrategy,
        biasInitializationStrategy,
        activationFunction,
        optimizationStrategy)

fun createMultiOutputEncoder(
    name : String?,
    numberSteps : Int,
    inputDimension : Int,
    hiddenDimension: Int,
    inputProjectionInitializationStrategy: InitializationStrategy,
    previousStateProjectionInitializationStrategy: InitializationStrategy,
    biasInitializationStrategy: InitializationStrategy?,
    activationFunction : ActivationFunction,
    optimizationStrategy : OptimizationStrategy? = null): MultiOutputDecoder {

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

    val encoder = MultiOutputDecoder(
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

package shape.komputation.layers.recurrent

import shape.komputation.functions.activation.ActivationFunction
import shape.komputation.functions.extractStep
import shape.komputation.initialization.InitializationStrategy
import shape.komputation.layers.ContinuationLayer
import shape.komputation.layers.OptimizableLayer
import shape.komputation.layers.feedforward.activation.createActivationLayers
import shape.komputation.matrix.DoubleMatrix
import shape.komputation.matrix.EMPTY_SEQUENCE_MATRIX
import shape.komputation.matrix.doubleZeroColumnVector
import shape.komputation.matrix.zeroSequenceMatrix
import shape.komputation.optimization.OptimizationStrategy
import java.util.*

class Decoder(
    name : String?,
    private val numberSteps : Int,
    private val inputDimension: Int,
    private val hiddenDimension : Int,
    private val numberCategories : Int,
    private val previousOutputProjection: SeriesProjection,
    private val stateProjection: SeriesProjection,
    private val stateActivations: Array<ContinuationLayer>,
    private val outputProjection: SeriesProjection,
    private val outputActivations: Array<ContinuationLayer>,
    private val bias : SeriesBias?) : ContinuationLayer(name), OptimizableLayer {

    private var previousOutput = doubleZeroColumnVector(hiddenDimension)
    private var state: DoubleMatrix = EMPTY_SEQUENCE_MATRIX

    override fun forward(input: DoubleMatrix): DoubleMatrix {

        Arrays.fill(previousOutput.entries, 0.0)

        this.state = input

        val seriesOutput = zeroSequenceMatrix(numberSteps, inputDimension)

        for (indexStep in 0..numberSteps - 1) {

            // Project the output from the previous step (but do nothing during step 1)
            val projectedPreviousOutput = previousOutputProjection.forwardStep(indexStep, this.previousOutput)
            val projectedPreviousOutputEntries = projectedPreviousOutput.entries

            // Project the previous state. At step 1, this is the encoding.
            val projectedState =  stateProjection.forwardStep(indexStep, this.state)
            val projectedStateEntries = projectedState.entries

            // Add the two projections
            val additionEntries = DoubleArray(hiddenDimension) { index ->

                projectedPreviousOutputEntries[index] + projectedStateEntries[index]

            }

            // Add the bias (if there is one)
            val statePreActivation =

                if(this.bias == null) {

                    additionEntries

                }
                else {

                    this.bias.forwardStep(additionEntries)

                }

            // Apply the activation function
            this.state = stateActivations[indexStep].forward(DoubleMatrix(hiddenDimension, 1, statePreActivation))

            // Project the state
            val outputPreActivation = outputProjection.forwardStep(indexStep, this.state)

            // Apply the activation function to the output pre-activation
            val stepOutput = outputActivations[indexStep].forward(outputPreActivation)

            // Store the set output in the series output
            seriesOutput.setStep(indexStep, stepOutput.entries)

            // Set the "previous output" at step t+1 to be the output at step t
            this.previousOutput = stepOutput

        }

        return seriesOutput

    }

    // Incoming gradient: d chain / d series prediction
    override fun backward(chain: DoubleMatrix): DoubleMatrix {

        val chainEntries = chain.entries

        var backwardStatePreActivationWrtPreviousOutput : DoubleMatrix? = null
        var backwardStatePreActivationWrtPreviousState : DoubleMatrix? = null

        for (indexStep in this.numberSteps - 1 downTo 0) {

            val stepChain = extractStep(chainEntries, indexStep, numberCategories)

            // Is this the last step?
            val isLastStep = indexStep == this.numberSteps - 1

            val outputSum =

                if (isLastStep) {

                    stepChain

                }
                else {

                    // Unless it is the last step, add the gradient with regard to the input from the step t+1 (which is the output of step t)
                    // d chain / d output(index+1) * d output(index+1) / d input(index + 1) *  d input(index + 1) / d output(index)
                    val backwardStatePreactivationWrtPreviousOutputEntries = backwardStatePreActivationWrtPreviousOutput!!.entries

                    DoubleArray(numberCategories) { index ->

                        stepChain[index] + backwardStatePreactivationWrtPreviousOutputEntries[index]

                    }

                }

            // Differentiate w.r.t. the output pre-activation
            // d output / d output pre-activation = d activate(output weights * state) / d output weights * state
            val backwardOutputWrtPreActivation = this.outputActivations[indexStep].backward(DoubleMatrix(numberCategories, 1, outputSum))

            // Differentiate w.r.t. the state
            // d output pre-activation (Wh) / d state = d output weights * state / d state
            val backwardOutputPreActivationWrtState = this.outputProjection.backwardStep(indexStep, backwardOutputWrtPreActivation)

            val stateSum =
                if (isLastStep) {

                    backwardOutputPreActivationWrtState.entries

                }
                else {

                    val backwardOutputPreActivationWrtStateEntries = backwardOutputPreActivationWrtState.entries

                    // d chain / d output(index+1) * d output(index+1) / d state(index)
                    val backwardStatePreActivationWrtPreviousStateEntries = backwardStatePreActivationWrtPreviousState!!.entries

                    DoubleArray(hiddenDimension) { index ->

                        backwardOutputPreActivationWrtStateEntries[index] + backwardStatePreActivationWrtPreviousStateEntries[index]

                    }

                }

            // Differentiate w.r.t. the state pre-activation
            // d state / d state pre-activation = d activate(state weights * state(index-1) + previous output weights * output(index-1) + bias) / d state weights * state(index-1) + previous output weights * output(index-1)
            val backwardStateWrtStatePreActivation = this.stateActivations[indexStep].backward(DoubleMatrix(hiddenDimension, 1, stateSum))

            // Differentiate w.r.t. the previous state
            // d state pre-activation / d previous state = d [ state weights * state(index-1) + previous output weights * output(index-1) + bias ] / d state(index-1)
            backwardStatePreActivationWrtPreviousState = this.stateProjection.backwardStep(indexStep, backwardStateWrtStatePreActivation)

            // Differentiate w.r.t. the previous output
            // d state pre-activation / d previous output = d [ state weights * state(index-1) + previous output weights * output(index-1) + bias ] / d output(index-1)
            backwardStatePreActivationWrtPreviousOutput = this.previousOutputProjection.backwardStep(indexStep, backwardStateWrtStatePreActivation)

            if (this.bias != null) {

                // Differentiate w.r.t. the bias
                this.bias.backwardStep(backwardStateWrtStatePreActivation)

            }

        }

        this.outputProjection.backwardSeries()
        this.stateProjection.backwardSeries()
        this.previousOutputProjection.backwardSeries()

        if (this.bias != null) {

            this.bias.backwardSeries()

        }

        return backwardStatePreActivationWrtPreviousState!!

    }

    override fun optimize() {

        this.outputProjection.optimize()
        this.stateProjection.optimize()
        this.previousOutputProjection.optimize()

        if (this.bias != null) {

            this.bias.optimize()

        }

    }

}

fun createDecoder(
    numberSteps: Int,
    inputDimension: Int,
    hiddenDimension: Int,
    numberCategories: Int,
    previousOutputProjectionInitializationStrategy: InitializationStrategy,
    previousStateProjectionInitializationStrategy: InitializationStrategy,
    biasInitializationStrategy: InitializationStrategy?,
    stateActivation: ActivationFunction,
    outputProjectionInitializationStrategy: InitializationStrategy,
    outputActivation: ActivationFunction,
    optimizationStrategy: OptimizationStrategy) =

    createDecoder(
        null,
        numberSteps,
        inputDimension,
        hiddenDimension,
        numberCategories,
        previousOutputProjectionInitializationStrategy,
        previousStateProjectionInitializationStrategy,
        biasInitializationStrategy,
        stateActivation,
        outputProjectionInitializationStrategy,
        outputActivation,
        optimizationStrategy)


fun createDecoder(
    name : String?,
    numberSteps: Int,
    inputDimension: Int,
    hiddenDimension: Int,
    numberCategories: Int,
    previousOutputProjectionInitializationStrategy: InitializationStrategy,
    previousStateProjectionInitializationStrategy: InitializationStrategy,
    biasInitializationStrategy: InitializationStrategy?,
    stateActivationFunction: ActivationFunction,
    outputProjectionInitializationStrategy: InitializationStrategy,
    outputActivationFunction: ActivationFunction,
    optimizationStrategy: OptimizationStrategy): Decoder {

    val previousOutputProjectionName = if(name == null) null else "$name-previous-output-projection"
    val previousOutputProjection = createSeriesProjection(previousOutputProjectionName, numberSteps, true, hiddenDimension, numberCategories, previousOutputProjectionInitializationStrategy, optimizationStrategy)

    val previousStateProjectionName = if(name == null) null else "$name-previous-output-projection"
    val previousStateProjection = createSeriesProjection(previousStateProjectionName, numberSteps, false, hiddenDimension, hiddenDimension, previousStateProjectionInitializationStrategy, optimizationStrategy)

    val stateActivationName = if(name == null) null else "$name-state-activation"
    val stateActivation = createActivationLayers(stateActivationName, numberSteps, stateActivationFunction)

    val outputProjectionName = if(name == null) null else "$name-output-projection"
    val outputProjection = createSeriesProjection(outputProjectionName, numberSteps, false, numberCategories, hiddenDimension, outputProjectionInitializationStrategy, optimizationStrategy)

    val outputActivationName = if(name == null) null else "$name-output-activation"
    val outputActivation = createActivationLayers(outputActivationName, numberSteps, outputActivationFunction)

    val bias =

        if(biasInitializationStrategy == null)
            null
        else
            createSeriesBias(if(name == null) null else "$name-bias", hiddenDimension, biasInitializationStrategy, optimizationStrategy)

    val decoder = Decoder(
        name,
        numberSteps,
        inputDimension,
        hiddenDimension,
        numberCategories,
        previousOutputProjection,
        previousStateProjection,
        stateActivation,
        outputProjection,
        outputActivation,
        bias)

    return decoder

}
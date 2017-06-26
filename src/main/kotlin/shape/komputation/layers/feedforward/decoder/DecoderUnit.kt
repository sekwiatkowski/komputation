package shape.komputation.layers.feedforward.decoder

import shape.komputation.functions.activation.ActivationFunction
import shape.komputation.functions.add
import shape.komputation.initialization.InitializationStrategy
import shape.komputation.layers.OptimizableLayer
import shape.komputation.layers.combination.AdditionCombination
import shape.komputation.layers.concatenateNames
import shape.komputation.layers.feedforward.activation.ActivationLayer
import shape.komputation.layers.feedforward.activation.createActivationLayers
import shape.komputation.layers.feedforward.projection.SeriesBias
import shape.komputation.layers.feedforward.projection.SeriesWeighting
import shape.komputation.layers.feedforward.projection.createSeriesBias
import shape.komputation.layers.feedforward.projection.createSeriesWeighting
import shape.komputation.matrix.DoubleMatrix
import shape.komputation.optimization.OptimizationStrategy

class DecoderUnit(
    private val name : String?,
    private val hiddenDimension : Int,
    private val outputDimension: Int,
    private val inputWeighting: SeriesWeighting,
    private val stateWeighting: SeriesWeighting,
    private val additions: Array<AdditionCombination>,
    private val stateActivation : Array<ActivationLayer>,
    private val outputWeighting: SeriesWeighting,
    private val outputActivations: Array<ActivationLayer>,
    private val bias : SeriesBias?) : OptimizableLayer {

    fun forward(step : Int, state : DoubleMatrix, input: DoubleMatrix): Pair<DoubleMatrix, DoubleMatrix> {

        // Weigh input
        // In the case of a single-input decoder, this is the t-1-th output.
        // In the case of a multi-input decoder, this is the t-th encoding.
        val weightedInput = this.inputWeighting.forwardStep(step, input)

        // Weigh the previous state.
        // At te first step in the case of a single-input decoder, this is the encoding.
        // At the first step in the case of a multi-input decoder, this is a zero vector.
        val weightedState = this.stateWeighting.forwardStep(step, state)

        // Add the two weightings
        val addition = this.additions[step].forward(weightedInput, weightedState)

        // Add the bias (if there is one)
        val statePreActivation =

            if(this.bias == null) {

                addition

            }
            else {

                this.bias.forwardStep(addition)

            }

        // Apply the activation function
        val newState = this.stateActivation[step].forward(statePreActivation)

        // Weigh the state
        val outputPreActivation = this.outputWeighting.forwardStep(step, newState)

        // Apply the activation function to the output pre-activation
        val newOutput = this.outputActivations[step].forward(outputPreActivation)

        return newState to newOutput

    }

    fun backwardStep(
        step : Int,
        chainStep: DoubleMatrix,
        backwardStatePreActivationWrtPreviousState : DoubleMatrix?): Pair<DoubleMatrix, DoubleMatrix> {

        // Differentiate w.r.t. the output pre-activation:
        // d output / d output pre-activation = d activate(output weights * state) / d output weights * state
        val backwardOutputWrtOutputPreActivation = this.outputActivations[step].backward(chainStep)

        // Differentiate w.r.t. the state:
        // d output pre-activation (Wh) / d state = d output weights * state / d state
        val backwardOutputPreActivationWrtState = this.outputWeighting.backwardStep(step, backwardOutputWrtOutputPreActivation)

        val stateSumEntries =

            if (backwardStatePreActivationWrtPreviousState == null) {

                backwardOutputPreActivationWrtState.entries

            }
            else {

                // d chain / d output(index+1) * d output(index+1) / d state(index)
                add(backwardOutputPreActivationWrtState.entries, backwardStatePreActivationWrtPreviousState.entries)

            }

        val stateSum = DoubleMatrix(hiddenDimension, 1, stateSumEntries)

        // Differentiate w.r.t. the state pre-activation:
        // d state / d state pre-activation = d activate(state weights * state(index-1) + previous output weights * output(index-1) + bias) / d state weights * state(index-1) + previous output weights * output(index-1)
        val backwardStateWrtStatePreActivation = this.stateActivation[step].backward(stateSum)

        // Differentiate w.r.t. the previous output:
        // d state pre-activation / d previous output = d [ state weights * state(index-1) + previous output weights * output(index-1) + bias ] / d output(index-1)
        val newBackwardStatePreActivationWrtInput = this.inputWeighting.backwardStep(step, backwardStateWrtStatePreActivation)

        // Differentiate w.r.t. the previous state:
        // d state pre-activation / d previous state = d [ state weights * state(index-1) + previous output weights * output(index-1) + bias ] / d state(index-1)
        val newBackwardStatePreActivationWrtPreviousState = this.stateWeighting.backwardStep(step, backwardStateWrtStatePreActivation)

        // Differentiate w.r.t. the bias
        this.bias?.backwardStep(backwardStateWrtStatePreActivation)

        return newBackwardStatePreActivationWrtInput to newBackwardStatePreActivationWrtPreviousState

    }

    fun backwardSeries() {

        this.outputWeighting.backwardSeries()
        this.stateWeighting.backwardSeries()
        this.inputWeighting.backwardSeries()

        this.bias?.backwardSeries()

    }

    override fun optimize() {

        this.outputWeighting.optimize()
        this.stateWeighting.optimize()
        this.inputWeighting.optimize()

        this.bias?.optimize()

    }

}

fun createDecoderUnit(
    numberSteps: Int,
    inputDimension : Int,
    hiddenDimension: Int,
    outputDimension: Int,
    useIdentityOnFirstInputWeighting : Boolean,
    previousOutputWeightingInitializationStrategy: InitializationStrategy,
    previousStateWeightingInitializationStrategy: InitializationStrategy,
    biasInitializationStrategy: InitializationStrategy?,
    stateActivation: ActivationFunction,
    outputWeightingInitializationStrategy: InitializationStrategy,
    outputActivation: ActivationFunction,
    optimizationStrategy: OptimizationStrategy?) =

    createDecoderUnit(
        null,
        numberSteps,
        inputDimension,
        hiddenDimension,
        outputDimension,
        useIdentityOnFirstInputWeighting,
        previousOutputWeightingInitializationStrategy,
        previousStateWeightingInitializationStrategy,
        biasInitializationStrategy,
        stateActivation,
        outputWeightingInitializationStrategy,
        outputActivation,
        optimizationStrategy)


fun createDecoderUnit(
    name : String?,
    numberSteps: Int,
    inputDimension : Int,
    hiddenDimension: Int,
    outputDimension: Int,
    useIdentityOnFirstInputWeighting : Boolean,
    inputWeightingInitializationStrategy: InitializationStrategy,
    previousStateWeightingInitializationStrategy: InitializationStrategy,
    biasInitializationStrategy: InitializationStrategy?,
    stateActivationFunction: ActivationFunction,
    outputWeightingInitializationStrategy: InitializationStrategy,
    outputActivationFunction: ActivationFunction,
    optimizationStrategy: OptimizationStrategy?): DecoderUnit {

    val inputWeightingName = concatenateNames(name, "input-weighting")
    val inputWeightingStepName = concatenateNames(name, "input-weighting-step")
    val inputWeighting = createSeriesWeighting(inputWeightingName, inputWeightingStepName, numberSteps, useIdentityOnFirstInputWeighting, inputDimension, hiddenDimension, inputWeightingInitializationStrategy, optimizationStrategy)

    val previousStateWeightingName = concatenateNames(name, "previous-state-weighting")
    val previousStateWeightingStepName = concatenateNames(name, "previous-state-weighting-step")
    val previousStateWeighting = createSeriesWeighting(previousStateWeightingName, previousStateWeightingStepName, numberSteps, false, hiddenDimension, hiddenDimension, previousStateWeightingInitializationStrategy, optimizationStrategy)

    val stateActivationName = concatenateNames(name, "state-activation-step")
    val stateActivations = createActivationLayers(numberSteps, stateActivationName, stateActivationFunction)

    val bias =

        if(biasInitializationStrategy == null)
            null
        else {

            val biasName = concatenateNames(name, "bias")
            createSeriesBias(biasName, hiddenDimension, biasInitializationStrategy, optimizationStrategy)

        }

    val additions = Array(numberSteps) { indexStep ->

        val additionName = concatenateNames(name, "addition-step-$indexStep")
        AdditionCombination(additionName)

    }

    val outputWeightingSeriesName = concatenateNames(name, "output-weighting")
    val outputWeightingStepName = concatenateNames(name, "output-weighting-step")
    val outputWeighting = createSeriesWeighting(outputWeightingSeriesName, outputWeightingStepName, numberSteps, false, hiddenDimension, outputDimension, outputWeightingInitializationStrategy, optimizationStrategy)

    val outputActivationName = concatenateNames(name, "output-activation-step")
    val outputActivations = createActivationLayers(numberSteps, outputActivationName, outputActivationFunction)

    val unit = DecoderUnit(
        name,
        hiddenDimension,
        outputDimension,
        inputWeighting,
        previousStateWeighting,
        additions,
        stateActivations,
        outputWeighting,
        outputActivations,
        bias
    )

    return unit

}
package shape.komputation.layers.feedforward.units

import shape.komputation.functions.activation.ActivationFunction
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

class SimpleRecurrentUnit(
    name : String?,
    private val inputWeighting: SeriesWeighting,
    private val previousStateWeighting: SeriesWeighting,
    private val additions : Array<AdditionCombination>,
    private val bias : SeriesBias?,
    private val activations: Array<ActivationLayer>) : RecurrentUnit(name), OptimizableLayer {

    override fun forwardStep(step : Int, state: DoubleMatrix, input: DoubleMatrix): DoubleMatrix {

        // weighted input = input weights * input
        val weightedInput = this.inputWeighting.forwardStep(step, input)

        // weighted state = state weights * state
        val weightedState =  this.previousStateWeighting.forwardStep(step, state)

        // addition = weighted input + weighted state
        val additionEntries = this.additions[step].forward(weightedInput, weightedState)

        // pre-activation = addition + bias
        val preActivation =

            if(this.bias == null) {

                additionEntries

            }
            else {

                this.bias.forwardStep(additionEntries)

            }


        // activation = activate(pre-activation)
        val newState = this.activations[step].forward(preActivation)

        return newState

    }

    override fun backwardStep(step : Int, chain: DoubleMatrix): Pair<DoubleMatrix, DoubleMatrix> {

        // d new state / state pre-activation
        // d activate(state weights * state(1) + input weights * input(2) + bias)) / d state weights * state(1) + input weights * input(2) + bias
        val backwardStateWrtStatePreActivation = this.activations[step].backward(chain)

        // d state weights * state(1) + input weights * input(2) + bias / d state(1) = state weights
        val backwardStatePreActivationWrtPreviousState = this.previousStateWeighting.backwardStep(step, backwardStateWrtStatePreActivation)

        // d state weights * state(1) + input weights * input(2) + bias / d input(2) = input weights
        val backwardStatePreActivationWrtInput = this.inputWeighting.backwardStep(step, backwardStateWrtStatePreActivation)

        // d state weights * state(1) + input weights * input(2) + bias / d bias = 1
        this.bias?.backwardStep(backwardStateWrtStatePreActivation)

        return backwardStatePreActivationWrtPreviousState to backwardStatePreActivationWrtInput

    }

    override fun backwardSeries() {

        this.previousStateWeighting.backwardSeries()
        this.inputWeighting.backwardSeries()

        this.bias?.backwardSeries()

    }

    override fun optimize() {

        this.previousStateWeighting.optimize()
        this.inputWeighting.optimize()

        this.bias?.optimize()

    }

}

fun createSimpleRecurrentUnit(
    numberSteps : Int,
    numberStepRows : Int,
    hiddenDimension: Int,
    stateWeightInitializationStrategy: InitializationStrategy,
    inputWeightInitializationStrategy: InitializationStrategy,
    biasInitializationStrategy: InitializationStrategy?,
    activationFunction : ActivationFunction,
    optimizationStrategy : OptimizationStrategy? = null) =

   createSimpleRecurrentUnit(
        null,
        numberSteps,
        numberStepRows,
        hiddenDimension,
        stateWeightInitializationStrategy,
        inputWeightInitializationStrategy,
        biasInitializationStrategy,
        activationFunction,
        optimizationStrategy)

fun createSimpleRecurrentUnit(
    name : String?,
    numberSteps : Int,
    inputDimension : Int,
    hiddenDimension: Int,
    previousStateWeightingInitializationStrategy: InitializationStrategy,
    inputWeightingInitializationStrategy: InitializationStrategy,
    biasInitializationStrategy: InitializationStrategy?,
    activationFunction : ActivationFunction,
    optimizationStrategy : OptimizationStrategy? = null): RecurrentUnit {

    val previousStateWeightingSeriesName = concatenateNames(name, "previous-state-weighting")
    val previousStateWeightingStepName = concatenateNames(name, "previous-state-weighting-step")

    val previousStateWeighting = createSeriesWeighting(
        previousStateWeightingSeriesName,
        previousStateWeightingStepName,
        numberSteps,
        true,
        hiddenDimension, hiddenDimension,
        previousStateWeightingInitializationStrategy,
        optimizationStrategy)

    val inputWeightingSeriesName = concatenateNames(name, "input-weighting")
    val inputWeightingStepName = concatenateNames(name, "input-weighting-step")

    val inputWeighting = createSeriesWeighting(
        inputWeightingSeriesName,
        inputWeightingStepName,
        numberSteps,
        false,
        inputDimension,
        hiddenDimension,
        inputWeightingInitializationStrategy,
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

    val activationName = concatenateNames(name, "activation")
    val activationLayers = createActivationLayers(numberSteps, activationName, activationFunction)

    val unitName = concatenateNames(name, "unit")
    val unit = SimpleRecurrentUnit(unitName, inputWeighting, previousStateWeighting, additions, bias, activationLayers)

    return unit

}
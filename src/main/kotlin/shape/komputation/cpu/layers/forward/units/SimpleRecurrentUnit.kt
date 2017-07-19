package shape.komputation.cpu.layers.forward.units

import shape.komputation.cpu.layers.combination.AdditionCombination
import shape.komputation.cpu.layers.combination.additionCombination
import shape.komputation.cpu.layers.forward.activation.CpuActivationLayer
import shape.komputation.cpu.layers.forward.activation.cpuActivationLayer
import shape.komputation.cpu.layers.forward.projection.SeriesBias
import shape.komputation.cpu.layers.forward.projection.SeriesWeighting
import shape.komputation.cpu.layers.forward.projection.seriesBias
import shape.komputation.cpu.layers.forward.projection.seriesWeighting
import shape.komputation.initialization.InitializationStrategy
import shape.komputation.layers.concatenateNames
import shape.komputation.layers.forward.activation.ActivationFunction
import shape.komputation.matrix.DoubleMatrix
import shape.komputation.optimization.Optimizable
import shape.komputation.optimization.OptimizationInstruction

class SimpleRecurrentUnit internal constructor(
    name: String?,
    private val previousStateWeighting: SeriesWeighting,
    private val inputWeighting: SeriesWeighting,
    private val additions: Array<AdditionCombination>,
    private val bias: SeriesBias?,
    private val activations: Array<CpuActivationLayer>) : RecurrentUnit(name), Optimizable {

    override fun forwardStep(step : Int, state: DoubleMatrix, input: DoubleMatrix, isTraining : Boolean): DoubleMatrix {

        // weighted state = state weights * state
        val weightedState = this.previousStateWeighting.forwardStep(step, state, isTraining)

        // weighted input = input weights * input
        val weightedInput = this.inputWeighting.forwardStep(step, input, isTraining)

        // addition = weighted input + weighted state
        val additionEntries = this.additions[step].forward(weightedState, weightedInput)

        // pre-activation = addition + bias
        val preActivation =

            if(this.bias == null) {

                additionEntries

            }
            else {

                this.bias.forwardStep(additionEntries)

            }


        // activation = activate(pre-activation)
        val newState = this.activations[step].forward(preActivation, isTraining)

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

    override fun optimize(scalingFactor : Double) {

        this.previousStateWeighting.optimize(scalingFactor)
        this.inputWeighting.optimize(scalingFactor)

        this.bias?.optimize(scalingFactor)

    }

}

fun simpleRecurrentUnit(
    numberSteps: Int,
    hiddenDimension: Int,
    inputDimension: Int,
    stateWeightInitializationStrategy: InitializationStrategy,
    inputWeightInitializationStrategy: InitializationStrategy,
    biasInitializationStrategy: InitializationStrategy?,
    activationFunction: ActivationFunction,
    optimizationStrategy: OptimizationInstruction? = null) =

    simpleRecurrentUnit(
        null,
        numberSteps,
        hiddenDimension,
        inputDimension,
        stateWeightInitializationStrategy,
        inputWeightInitializationStrategy,
        biasInitializationStrategy,
        activationFunction,
        optimizationStrategy)

fun simpleRecurrentUnit(
    name: String?,
    numberSteps: Int,
    hiddenDimension: Int,
    inputDimension: Int,
    previousStateWeightingInitializationStrategy: InitializationStrategy,
    inputWeightingInitializationStrategy: InitializationStrategy,
    biasInitializationStrategy: InitializationStrategy?,
    activationFunction: ActivationFunction,
    optimizationStrategy: OptimizationInstruction? = null): RecurrentUnit {

    val previousStateWeightingSeriesName = concatenateNames(name, "previous-state-weighting")
    val previousStateWeightingStepName = concatenateNames(name, "previous-state-weighting-step")

    val previousStateWeighting = seriesWeighting(
        previousStateWeightingSeriesName,
        previousStateWeightingStepName,
        numberSteps,
        true,
        hiddenDimension,
        hiddenDimension,
        previousStateWeightingInitializationStrategy,
        optimizationStrategy)

    val inputWeightingSeriesName = concatenateNames(name, "input-weighting")
    val inputWeightingStepName = concatenateNames(name, "input-weighting-step")

    val inputWeighting = seriesWeighting(
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
        additionCombination(additionName)

    }

    val bias =

        if(biasInitializationStrategy == null)
            null
        else
            seriesBias(concatenateNames(name, "bias"), hiddenDimension, biasInitializationStrategy, optimizationStrategy)

    val activationName = concatenateNames(name, "activation")
    val activationLayers = Array(numberSteps) { index ->

        cpuActivationLayer(concatenateNames(activationName, index.toString()), activationFunction, hiddenDimension, numberSteps).buildForCpu()

    }

    val unitName = concatenateNames(name, "unit")
    val unit = SimpleRecurrentUnit(unitName, previousStateWeighting, inputWeighting, additions, bias, activationLayers)

    return unit

}
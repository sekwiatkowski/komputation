package shape.komputation.layers.forward.units

import shape.komputation.initialization.InitializationStrategy
import shape.komputation.layers.combination.AdditionCombination
import shape.komputation.layers.combination.HadamardCombination
import shape.komputation.layers.combination.additionCombination
import shape.komputation.layers.combination.hadamardCombination
import shape.komputation.layers.concatenateNames
import shape.komputation.layers.forward.CounterProbabilityLayer
import shape.komputation.layers.forward.activation.ActivationLayer
import shape.komputation.layers.forward.activation.sigmoidLayer
import shape.komputation.layers.forward.activation.tanhLayer
import shape.komputation.layers.forward.counterProbabilityLayer
import shape.komputation.layers.forward.projection.seriesBias
import shape.komputation.layers.forward.projection.seriesWeighting
import shape.komputation.matrix.DoubleMatrix
import shape.komputation.matrix.doubleColumnVector
import shape.komputation.optimization.DenseAccumulator
import shape.komputation.optimization.Optimizable
import shape.komputation.optimization.OptimizationStrategy

class MinimalGatedUnit internal constructor(
    name: String?,
    hiddenDimension: Int,
    inputDimension: Int,
    private val forgetUnit: RecurrentUnit,
    private val shortTermResponse: ShortTermResponse,
    private val counterProbabilities: Array<CounterProbabilityLayer>,
    private val longTermHadamards: Array<HadamardCombination>,
    private val shortTermHadamards: Array<HadamardCombination>,
    private val stateAdditions: Array<AdditionCombination>) : RecurrentUnit(name), Optimizable {

    private val previousStateAccumulator = DenseAccumulator(hiddenDimension)
    private val inputAccumulator = DenseAccumulator(inputDimension)

    override fun forwardStep(step : Int, state : DoubleMatrix, input : DoubleMatrix, isTraining : Boolean): DoubleMatrix {

        val forget = this.forgetUnit.forwardStep(step, state, input, isTraining)

        val oneMinusForget = this.counterProbabilities[step].forward(forget, isTraining)

        val longTermComponent = this.longTermHadamards[step].forward(oneMinusForget, state)

        val shortTermResponse = this.shortTermResponse.forward(step, state, input, forget, isTraining)

        val shortTermComponent = this.shortTermHadamards[step].forward(forget, shortTermResponse)

        val newState = this.stateAdditions[step].forward(longTermComponent, shortTermComponent)

        return newState
    }


    private fun backwardLongTermComponent(step: Int, diffChainWrtLongTermComponent: DoubleMatrix) {

        // (1 - forget) (.) previous state / d (1 - forget) = previous state
        val diffLongTermComponentWrtKeep = this.longTermHadamards[step].backwardFirst(diffChainWrtLongTermComponent)

        // d (1 - forget) / d forget = -1
        val diffKeepWrtForget = this.counterProbabilities[step].backward(diffLongTermComponentWrtKeep)
        val (diffForgetWrtPreviousState, diffForgetWrtInput) = this.forgetUnit.backwardStep(step, diffKeepWrtForget)

        this.previousStateAccumulator.accumulate(diffForgetWrtPreviousState.entries)
        this.inputAccumulator.accumulate(diffForgetWrtInput.entries)

        // (1 - forget) (.) previous state / d previous state = (1 - forget)
        val diffLongTermComponentWrtPreviousState = this.longTermHadamards[step].backwardSecond(diffChainWrtLongTermComponent)
        this.previousStateAccumulator.accumulate(diffLongTermComponentWrtPreviousState.entries)
    }

    private fun backwardShortTermComponent(step: Int, diffChainWrtShortTermComponent: DoubleMatrix) {

        // short-term component = forget (.) short-term response

        // d short-term component / forget = short-term response
        val diffShortTermComponentWrtForget = this.shortTermHadamards[step].backwardFirst(diffChainWrtShortTermComponent)

        // d forget / d previous state, d forget / input
        val (diffShortTermComponentForgetWrtPreviousState, diffShortTermComponentForgetWrtInput) = this.forgetUnit.backwardStep(step, diffShortTermComponentWrtForget)
        this.previousStateAccumulator.accumulate(diffShortTermComponentForgetWrtPreviousState.entries)
        this.inputAccumulator.accumulate(diffShortTermComponentForgetWrtInput.entries)

        // d short-term component / short-term response = forget
        val diffShortTermComponentWrtShortTermResponse = this.shortTermHadamards[step].backwardSecond(diffChainWrtShortTermComponent)

        val (diffShortTermMemoryWrtForget, shortTermResponsePair) = this.shortTermResponse.backward(step, diffShortTermComponentWrtShortTermResponse)
        val (diffShortTermMemoryWrtPreviousState, diffShortTermWeightedInputWrtInput) = shortTermResponsePair

        // d forget / d previous state, d forget / input
        val (diffShortTermMemoryForgetWrtPreviousState, diffShortTermMemoryForgetWrtInput) = this.forgetUnit.backwardStep(step, diffShortTermMemoryWrtForget)
        this.previousStateAccumulator.accumulate(diffShortTermMemoryForgetWrtPreviousState.entries)
        this.inputAccumulator.accumulate(diffShortTermMemoryForgetWrtInput.entries)

        this.previousStateAccumulator.accumulate(diffShortTermMemoryWrtPreviousState.entries)
        this.inputAccumulator.accumulate(diffShortTermWeightedInputWrtInput.entries)

    }

    override fun backwardStep(step : Int, chain : DoubleMatrix): Pair<DoubleMatrix, DoubleMatrix> {

        // d (long-term component + short-term component) / d long-term component
        val diffChainWrtLongTermComponent = this.stateAdditions[step].backwardFirst(chain)
        this.backwardLongTermComponent(step, diffChainWrtLongTermComponent)

        // d (long-term component + short-term component) / d short-term component
        val diffChainWrtShortTermComponent = this.stateAdditions[step].backwardSecond(chain)
        this.backwardShortTermComponent(step, diffChainWrtShortTermComponent)

        val previousStateAccumulation = doubleColumnVector(*this.previousStateAccumulator.getAccumulation().copyOf())
        val inputAccumulation = doubleColumnVector(*this.inputAccumulator.getAccumulation().copyOf())

        this.inputAccumulator.reset()
        this.previousStateAccumulator.reset()

        return previousStateAccumulation to inputAccumulation

    }

    override fun backwardSeries() {

        this.forgetUnit.backwardSeries()
        this.shortTermResponse.backwardSeries()

    }

    override fun optimize() {

        if (this.forgetUnit is Optimizable) {

            this.forgetUnit.optimize()

        }

        this.shortTermResponse.optimize()

    }

}

fun minimalGatedUnit(
    numberSteps: Int,
    hiddenDimension: Int,
    inputDimension: Int,
    forgetPreviousStateWeightInitializationStrategy: InitializationStrategy,
    forgetInputWeightInitializationStrategy: InitializationStrategy,
    forgetBiasInitializationStrategy: InitializationStrategy?,
    shortTermMemoryWeightInitializationStrategy: InitializationStrategy,
    shortTermInputWeightInitializationStrategy: InitializationStrategy,
    shortTermBiasInitializationStrategy: InitializationStrategy?,
    optimizationStrategy: OptimizationStrategy? = null) =

    minimalGatedUnit(
        null,
        numberSteps,
        hiddenDimension,
        inputDimension,
        forgetPreviousStateWeightInitializationStrategy,
        forgetInputWeightInitializationStrategy,
        forgetBiasInitializationStrategy,
        shortTermMemoryWeightInitializationStrategy,
        shortTermInputWeightInitializationStrategy,
        shortTermBiasInitializationStrategy,
        optimizationStrategy)

fun minimalGatedUnit(
    name: String?,
    numberSteps: Int,
    hiddenDimension: Int,
    inputDimension: Int,
    forgetPreviousStateWeightInitializationStrategy: InitializationStrategy,
    forgetInputWeightInitializationStrategy: InitializationStrategy,
    forgetBiasInitializationStrategy: InitializationStrategy?,
    shortTermMemoryWeightInitializationStrategy: InitializationStrategy,
    shortTermInputWeightInitializationStrategy: InitializationStrategy,
    shortTermBiasInitializationStrategy: InitializationStrategy?,
    optimizationStrategy: OptimizationStrategy? = null): RecurrentUnit {

    val forgetPreviousStateWeightingSeriesName = concatenateNames(name, "forget-previous-state-weighting")
    val forgetPreviousStateWeightingStepName = concatenateNames(name, "forget-previous-state-weighting-step")
    val forgetPreviousStateWeighting = seriesWeighting(forgetPreviousStateWeightingSeriesName, forgetPreviousStateWeightingStepName, numberSteps, true, hiddenDimension, hiddenDimension, forgetPreviousStateWeightInitializationStrategy, optimizationStrategy)

    val forgetInputWeightingSeriesName = concatenateNames(name, "forget-input-weighting")
    val forgetInputWeightingStepName = concatenateNames(name, "forget-input-weighting-step")
    val forgetInputWeighting = seriesWeighting(forgetInputWeightingSeriesName, forgetInputWeightingStepName, numberSteps, false, inputDimension, hiddenDimension, forgetInputWeightInitializationStrategy, optimizationStrategy)

    val forgetAdditions = Array(numberSteps) { indexStep ->

        val forgetAdditionName = concatenateNames(name, "forget-addition-step-$indexStep")
        additionCombination(forgetAdditionName)

    }

    val forgetBias =

        if (forgetBiasInitializationStrategy != null) {

            val forgetBiasName = concatenateNames(name, "forget-bias")
            seriesBias(forgetBiasName, hiddenDimension, forgetBiasInitializationStrategy, optimizationStrategy)

        }
        else {

            null

        }

    val forgetActivations = Array<ActivationLayer>(numberSteps) { indexStep ->

        val forgetActivationName = concatenateNames(name, "forget-activation-step-$indexStep")
        sigmoidLayer(forgetActivationName)

    }

    val forgetUnit = SimpleRecurrentUnit(name, forgetPreviousStateWeighting, forgetInputWeighting, forgetAdditions, forgetBias, forgetActivations)

    val shortTermResponse = shortTermResponse("short-term-response", numberSteps, hiddenDimension, inputDimension, shortTermMemoryWeightInitializationStrategy, shortTermInputWeightInitializationStrategy, shortTermBiasInitializationStrategy, optimizationStrategy)

    val keepSubtractions = Array(numberSteps) { indexStep ->

        val keepSubtractionName = concatenateNames(name, "keep-subtraction-$indexStep")
        counterProbabilityLayer(keepSubtractionName, hiddenDimension)

    }

    val shortTermHadamards = Array(numberSteps) { indexStep ->

        val shortTermHadamardName = concatenateNames(name, "short-term-hadamard-step-$indexStep")
        hadamardCombination(shortTermHadamardName)

    }

    val longTermHadamards = Array(numberSteps) { indexStep ->

        val shortTermHadamardName = concatenateNames(name, "long-term-hadamard-step-$indexStep")
        hadamardCombination(shortTermHadamardName)

    }

    val stateAdditions = Array(numberSteps) { indexStep ->

        val shortTermAdditionName = concatenateNames(name, "state-addition-step-$indexStep")
        additionCombination(shortTermAdditionName)

    }

    val minimalGatedUnit = MinimalGatedUnit(
        name,
        hiddenDimension,
        inputDimension,
        forgetUnit,
        shortTermResponse,
        keepSubtractions,
        shortTermHadamards,
        longTermHadamards,
        stateAdditions)

    return minimalGatedUnit

}
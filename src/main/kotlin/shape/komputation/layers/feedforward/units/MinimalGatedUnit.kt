package shape.komputation.layers.feedforward.units

import shape.komputation.initialization.InitializationStrategy
import shape.komputation.layers.OptimizableLayer
import shape.komputation.layers.combination.AdditionCombination
import shape.komputation.layers.combination.HadamardCombination
import shape.komputation.layers.combination.SubtractionCombination
import shape.komputation.layers.concatenateNames
import shape.komputation.layers.feedforward.activation.ActivationLayer
import shape.komputation.layers.feedforward.activation.SigmoidLayer
import shape.komputation.layers.feedforward.activation.TanhLayer
import shape.komputation.layers.feedforward.projection.createSeriesBias
import shape.komputation.layers.feedforward.projection.createSeriesWeighting
import shape.komputation.matrix.DoubleMatrix
import shape.komputation.matrix.doubleColumnVector
import shape.komputation.matrix.doubleOneColumnVector
import shape.komputation.optimization.DenseAccumulator
import shape.komputation.optimization.OptimizationStrategy

class MinimalGatedUnit(
    name : String?,
    inputDimension : Int,
    hiddenDimension : Int,
    private val forgetUnit : RecurrentUnit,
    private val shortTermResponse : ShortTermResponse,
    private val keepSubtractions : Array<SubtractionCombination>,
    private val longTermHadamards : Array<HadamardCombination>,
    private val shortTermHadamards : Array<HadamardCombination>,
    private val stateAdditions : Array<AdditionCombination>) : RecurrentUnit(name), OptimizableLayer {

    private val one = doubleOneColumnVector(hiddenDimension)

    private val previousStateAccumulator = DenseAccumulator(hiddenDimension)
    private val inputAccumulator = DenseAccumulator(inputDimension)

    override fun forwardStep(step : Int, state : DoubleMatrix, input : DoubleMatrix): DoubleMatrix {

        val forget = this.forgetUnit.forwardStep(step, state, input)

        val oneMinusForget = this.keepSubtractions[step].forward(this.one, forget)

        val longTermComponent = this.longTermHadamards[step].forward(oneMinusForget, state)

        val shortTermResponse = this.shortTermResponse.forward(step, state, input, forget)

        val shortTermComponent = this.shortTermHadamards[step].forward(forget, shortTermResponse)

        val newState = this.stateAdditions[step].forward(longTermComponent, shortTermComponent)

        return newState
    }


    private fun backwardLongTermComponent(step: Int, diffChainWrtLongTermComponent: DoubleMatrix) {

        // (1 - forget) (.) previous state / d (1 - forget) = previous state
        val diffLongTermComponentWrtKeep = this.longTermHadamards[step].backwardFirst(diffChainWrtLongTermComponent)

        // d (1 - forget) / d forget = -1
        val diffKeepWrtForget = this.keepSubtractions[step].backwardSecond(diffLongTermComponentWrtKeep)
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

        if (this.forgetUnit is OptimizableLayer) {

            this.forgetUnit.optimize()

        }

        this.shortTermResponse.optimize()

    }

}

fun createMinimalGatedUnit(
    numberSteps : Int,
    inputDimension : Int,
    hiddenDimension: Int,
    forgetPreviousStateWeightInitializationStrategy: InitializationStrategy,
    forgetInputWeightInitializationStrategy: InitializationStrategy,
    forgetBiasInitializationStrategy: InitializationStrategy?,
    shortTermMemoryWeightInitializationStrategy : InitializationStrategy,
    shortTermInputWeightInitializationStrategy : InitializationStrategy,
    shortTermBiasInitializationStrategy : InitializationStrategy?,
    optimizationStrategy : OptimizationStrategy? = null) =

    createMinimalGatedUnit(
        null,
        numberSteps,
        inputDimension,
        hiddenDimension,
        forgetPreviousStateWeightInitializationStrategy,
        forgetInputWeightInitializationStrategy,
        forgetBiasInitializationStrategy,
        shortTermMemoryWeightInitializationStrategy,
        shortTermInputWeightInitializationStrategy,
        shortTermBiasInitializationStrategy,
        optimizationStrategy)

fun createMinimalGatedUnit(
    name : String?,
    numberSteps : Int,
    inputDimension : Int,
    hiddenDimension: Int,
    forgetPreviousStateWeightInitializationStrategy: InitializationStrategy,
    forgetInputWeightInitializationStrategy: InitializationStrategy,
    forgetBiasInitializationStrategy: InitializationStrategy?,
    shortTermMemoryWeightInitializationStrategy : InitializationStrategy,
    shortTermInputWeightInitializationStrategy : InitializationStrategy,
    shortTermBiasInitializationStrategy : InitializationStrategy?,
    optimizationStrategy : OptimizationStrategy? = null): RecurrentUnit {

    val forgetPreviousStateWeightingSeriesName = concatenateNames(name, "forget-previous-state-weighting")
    val forgetPreviousStateWeightingStepName = concatenateNames(name, "forget-previous-state-weighting-step")
    val forgetPreviousStateWeighting = createSeriesWeighting(forgetPreviousStateWeightingSeriesName, forgetPreviousStateWeightingStepName, numberSteps, true, hiddenDimension, hiddenDimension, forgetPreviousStateWeightInitializationStrategy, optimizationStrategy)

    val forgetInputWeightingSeriesName = concatenateNames(name, "forget-input-weighting")
    val forgetInputWeightingStepName = concatenateNames(name, "forget-input-weighting-step")
    val forgetInputWeighting = createSeriesWeighting(forgetInputWeightingSeriesName, forgetInputWeightingStepName, numberSteps, false, inputDimension, hiddenDimension, forgetInputWeightInitializationStrategy, optimizationStrategy)

    val forgetAdditions = Array(numberSteps) { indexStep ->

        val forgetAdditionName = concatenateNames(name, "forget-addition-step-$indexStep")
        AdditionCombination(forgetAdditionName)

    }

    val forgetBias =

        if (forgetBiasInitializationStrategy != null) {

            val forgetBiasName = concatenateNames(name, "forget-bias")
            createSeriesBias(forgetBiasName, hiddenDimension, forgetBiasInitializationStrategy, optimizationStrategy)

        }
        else {

            null

        }

    val forgetActivations = Array<ActivationLayer>(numberSteps) { indexStep ->

        val forgetActivationName = concatenateNames(name, "forget-activation-step-$indexStep")
        SigmoidLayer(forgetActivationName)

    }

    val forgetUnit = SimpleRecurrentUnit(name, forgetPreviousStateWeighting, forgetInputWeighting, forgetAdditions, forgetBias, forgetActivations)

    val shortTermForgetting = Array(numberSteps) { indexStep ->

        val shortTermForgettingName = concatenateNames(name, "short-term-forgetting-step-$indexStep")
        HadamardCombination(shortTermForgettingName)

    }

    val shortTermMemoryWeightingSeriesName = concatenateNames(name, "short-term-memory-weighting")
    val shortTermMemoryWeightingStepName = concatenateNames(name, "short-term-memory-weighting-step")
    val shortTermMemoryWeighting = createSeriesWeighting(shortTermMemoryWeightingSeriesName, shortTermMemoryWeightingStepName, numberSteps, true, hiddenDimension, hiddenDimension, shortTermMemoryWeightInitializationStrategy, optimizationStrategy)

    val shortTermInputWeightingSeriesName = concatenateNames(name, "short-term-input-weighting")
    val shortTermInputWeightingStepName = concatenateNames(name, "short-term-input-weighting-step")
    val shortTermInputWeighting = createSeriesWeighting(shortTermInputWeightingSeriesName, shortTermInputWeightingStepName, numberSteps, false, inputDimension, hiddenDimension, shortTermInputWeightInitializationStrategy, optimizationStrategy)

    val shortTermAdditions = Array(numberSteps) { indexStep ->

        val shortTermAdditionName = concatenateNames(name, "short-term-addition-step-$indexStep")
        AdditionCombination(shortTermAdditionName)

    }

    val shortTermBias =

        if (shortTermBiasInitializationStrategy != null) {

            val shortTermBiasName = concatenateNames(name, "short-term-bias")
            createSeriesBias(shortTermBiasName, hiddenDimension, shortTermBiasInitializationStrategy, optimizationStrategy)
        }
        else {
            null
        }


    val shortTermActivations = Array(numberSteps) { indexStep ->

        val shortTermActivationName = concatenateNames(name, "short-term-activation-step-$indexStep")
        TanhLayer(shortTermActivationName)

    }

    val shortTermResponse = ShortTermResponse(shortTermForgetting, shortTermMemoryWeighting, shortTermInputWeighting, shortTermAdditions, shortTermBias, shortTermActivations)

    val keepSubtractions = Array(numberSteps) { indexStep ->

        val keepSubtractionName = concatenateNames(name, "keep-subtraction-$indexStep")
        SubtractionCombination(keepSubtractionName)

    }

    val shortTermHadamards = Array(numberSteps) { indexStep ->

        val shortTermHadamardName = concatenateNames(name, "short-term-hadamard-step-$indexStep")
        HadamardCombination(shortTermHadamardName)

    }

    val longTermHadamards = Array(numberSteps) { indexStep ->

        val shortTermHadamardName = concatenateNames(name, "short-term-hadamard-step-$indexStep")
        HadamardCombination(shortTermHadamardName)

    }

    val stateAdditions = Array(numberSteps) { indexStep ->

        val shortTermAdditionName = concatenateNames(name, "short-term-addition-step-$indexStep")
        AdditionCombination(shortTermAdditionName)

    }

    val minimalGatedUnit = MinimalGatedUnit(
        name,
        inputDimension,
        hiddenDimension,
        forgetUnit,
        shortTermResponse,
        keepSubtractions,
        shortTermHadamards,
        longTermHadamards,
        stateAdditions)

    return minimalGatedUnit

}
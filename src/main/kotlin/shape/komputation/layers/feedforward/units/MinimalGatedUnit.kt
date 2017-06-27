package shape.komputation.layers.feedforward.units

import shape.komputation.initialization.InitializationStrategy
import shape.komputation.layers.OptimizableLayer
import shape.komputation.layers.combination.AdditionCombination
import shape.komputation.layers.combination.HadamardCombination
import shape.komputation.layers.combination.SubtractionCombination
import shape.komputation.layers.concatenateNames
import shape.komputation.layers.feedforward.activation.SigmoidLayer
import shape.komputation.layers.feedforward.activation.TanhLayer
import shape.komputation.layers.feedforward.projection.SeriesBias
import shape.komputation.layers.feedforward.projection.SeriesWeighting
import shape.komputation.layers.feedforward.projection.createSeriesBias
import shape.komputation.layers.feedforward.projection.createSeriesWeighting
import shape.komputation.matrix.DoubleMatrix
import shape.komputation.matrix.doubleColumnVector
import shape.komputation.matrix.doubleOneColumnVector
import shape.komputation.matrix.doubleZeroColumnVector
import shape.komputation.optimization.DenseAccumulator
import shape.komputation.optimization.OptimizationStrategy

class MinimalGatedUnit(
    name : String?,
    inputDimension : Int,
    hiddenDimension : Int,
    private val forgetPreviousStateWeighting: SeriesWeighting,
    private val forgetInputWeighting: SeriesWeighting,
    private val forgetAdditions: Array<AdditionCombination>,
    private val forgetBias : SeriesBias?,
    private val forgetActivations: Array<SigmoidLayer>,
    private val shortTermForgetting : Array<HadamardCombination>,
    private val shortTermMemoryWeighting: SeriesWeighting,
    private val shortTermInputWeighting: SeriesWeighting,
    private val shortTermAdditions: Array<AdditionCombination>,
    private val shortTermBias : SeriesBias?,
    private val shortTermActivations: Array<TanhLayer>,
    private val keepSubtractions: Array<SubtractionCombination>,
    private val longTermHadamards: Array<HadamardCombination>,
    private val shortTermHadamards: Array<HadamardCombination>,
    private val stateAdditions: Array<AdditionCombination>) : RecurrentUnit(name), OptimizableLayer {

    private val one = doubleOneColumnVector(hiddenDimension)

    private val previousStateAccumulator = DenseAccumulator(hiddenDimension)
    private val inputAccumulator = DenseAccumulator(inputDimension)

    fun forwardForget(step : Int, state : DoubleMatrix, input : DoubleMatrix): DoubleMatrix {

        // forget weighted previous state = forget previous state weights * previous state
        val forgetWeightedPreviousState = this.forgetPreviousStateWeighting.forwardStep(step, state)

        // forget weighted input = forget input weights * input
        val forgetWeightedInput = this.forgetInputWeighting.forwardStep(step, input)

        // forget addition = forget weighted input + forget weighted previous state
        val forgetAddition = this.forgetAdditions[step].forward(forgetWeightedPreviousState, forgetWeightedInput)

        // forget pre-activation = forget addition (+ forget bias)
        val forgetPreActivation =

            if (this.forgetBias == null) {

                forgetAddition

            }
            else {

                this.forgetBias.forwardStep(forgetAddition)
            }

        // forget = sigmoid(forget pre-activation)
        val forget = this.forgetActivations[step].forward(forgetPreActivation)

        return forget
    }

    fun forwardShortTermResponse(step : Int, state : DoubleMatrix, input : DoubleMatrix, forget : DoubleMatrix): DoubleMatrix {

        val shortTermMemory = this.shortTermForgetting[step].forward(state, forget)

        val shortTermWeightedMemory = this.shortTermMemoryWeighting.forwardStep(step, shortTermMemory)

        val shortTermWeightedInput = this.shortTermInputWeighting.forwardStep(step, input)

        val shortTermAddition = this.shortTermAdditions[step].forward(shortTermWeightedMemory, shortTermWeightedInput)

        // forget pre-activation = forget addition (+ forget bias)
        val shortTermPreActivation =

            if (this.shortTermBias == null) {

                shortTermAddition

            }
            else {

                this.shortTermBias.forwardStep(shortTermAddition)
            }

        val shortTermResponse = this.shortTermActivations[step].forward(shortTermPreActivation)

        return shortTermResponse
    }

    override fun forwardStep(step : Int, state : DoubleMatrix, input : DoubleMatrix): DoubleMatrix {

        val forget = this.forwardForget(step, state, input)

        val oneMinusForget = this.keepSubtractions[step].forward(this.one, forget)

        val longTermComponent = this.longTermHadamards[step].forward(oneMinusForget, state)

        val shortTermResponse = this.forwardShortTermResponse(step, state, input, forget)

        val shortTermComponent = this.shortTermHadamards[step].forward(forget, shortTermResponse)

        val newState = this.stateAdditions[step].forward(longTermComponent, shortTermComponent)

        return newState
    }

    fun backwardWrtForget(step : Int, chain: DoubleMatrix) {

        // forget = sigmoid(forget pre-activation)

        // d forget / d forget pre-activation
        val diffForgetWrtPreActivation = this.forgetActivations[step].backward(chain)

        // forget pre-activation = forget addition (+ forget bias)
        // forget addition = forget projected input + forget projected previous state

        // d forget pre-activation / d forget addition = 1

        // d forget addition / d projected previous state
        val diffForgetPreActivationWrtProjectedPreviousState = this.forgetAdditions[step].backwardFirst(diffForgetWrtPreActivation)

        // d projected previous state / d previous state
        val diffForgetWeightedPreviousStateWrtPreviousState = this.forgetPreviousStateWeighting.backwardStep(step, diffForgetPreActivationWrtProjectedPreviousState)

        this.previousStateAccumulator.accumulate(diffForgetWeightedPreviousStateWrtPreviousState.entries)

        // d forget addition / d weighted input
        val diffForgetPreActivationWrtWeightedInput = this.forgetAdditions[step].backwardSecond(diffForgetWrtPreActivation)

        // d weighted input / d input
        val diffForgetWeightedInputWrtInput = this.forgetInputWeighting.backwardStep(step, diffForgetPreActivationWrtWeightedInput)

        this.inputAccumulator.accumulate(diffForgetWeightedInputWrtInput.entries)

        if (this.forgetBias != null) {

            // d forget addition / d projected input
            this.forgetBias.backwardStep(diffForgetWrtPreActivation)

        }

    }

    private fun backwardLongTermComponent(step: Int, diffChainWrtLongTermComponent: DoubleMatrix) {

        // (1 - forget) (.) previous state / d (1 - forget) = previous state
        val diffLongTermComponentWrtKeep = this.longTermHadamards[step].backwardFirst(diffChainWrtLongTermComponent)

        // d (1 - forget) / d forget = -1
        val diffKeepWrtForget = this.keepSubtractions[step].backwardSecond(diffLongTermComponentWrtKeep)

        this.backwardWrtForget(step, diffKeepWrtForget)

        // (1 - forget) (.) previous state / d previous state = (1 - forget)
        val diffLongTermComponentWrtPreviousState = this.longTermHadamards[step].backwardSecond(diffChainWrtLongTermComponent)

        this.previousStateAccumulator.accumulate(diffLongTermComponentWrtPreviousState.entries)
    }

    private fun backwardShortTermComponent(step: Int, diffChainWrtShortTermComponent: DoubleMatrix) {

        // short-term component = forget (.) short-term response

        // d short-term component / forget = short-term response
        val diffShortTermComponentWrtForget = this.shortTermHadamards[step].backwardFirst(diffChainWrtShortTermComponent)

        this.backwardWrtForget(step, diffShortTermComponentWrtForget)

        // d short-term component / short-term response = forget
        val diffShortTermComponentWrtShortTermResponse = this.shortTermHadamards[step].backwardSecond(diffChainWrtShortTermComponent)

        // short-term response = tanh(short-term response pre-activation)
        // d short-term response / d short-term response pre-activation
        val diffShortTermResponseWrtPreActivation = this.shortTermActivations[step].backward(diffShortTermComponentWrtShortTermResponse)

        // short-term response pre-activation = weighted short-term memory + weighted input (+ short-term bias)

        // d short-term response pre-activation / d weighted short-term memory
        val diffShortTermResponsePreActivationWrtWeightedShortTermMemory = this.shortTermAdditions[step].backwardFirst(diffShortTermResponseWrtPreActivation)

        // d weighted short-term memory / d short-term memory
        val diffWeightedShortTermMemoryWrtShortTermMemory = this.shortTermMemoryWeighting.backwardStep(step, diffShortTermResponsePreActivationWrtWeightedShortTermMemory)

        // d short-term memory / d forget
        val diffShortTermMemoryWrtForget = this.shortTermForgetting[step].backwardFirst(diffWeightedShortTermMemoryWrtShortTermMemory)

        this.backwardWrtForget(step, diffShortTermMemoryWrtForget)

        // d short-term memory / d previous state
        val diffShortTermMemoryWrtPreviousState = this.shortTermForgetting[step].backwardFirst(diffWeightedShortTermMemoryWrtShortTermMemory)
        this.previousStateAccumulator.accumulate(diffShortTermMemoryWrtPreviousState.entries)

        // d short-term response pre-activation / d short-term weighted input
        val diffShortTermResponsePreActivationWrtWeightedInput = this.shortTermAdditions[step].backwardSecond(diffShortTermResponseWrtPreActivation)

        // d short-term weighted input / d weighted input
        val diffShortTermWeightedInputWrtInput = this.shortTermInputWeighting.backwardStep(step, diffShortTermResponsePreActivationWrtWeightedInput)
        this.inputAccumulator.accumulate(diffShortTermWeightedInputWrtInput.entries)

        if (this.shortTermBias != null) {

            shortTermBias.backwardStep(diffShortTermResponseWrtPreActivation)

        }

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

        this.forgetPreviousStateWeighting.backwardSeries()
        this.forgetInputWeighting.backwardSeries()
        this.forgetBias?.backwardSeries()

        this.shortTermMemoryWeighting.backwardSeries()
        this.shortTermInputWeighting.backwardSeries()
        this.shortTermBias?.backwardSeries()

    }

    override fun optimize() {

        this.forgetPreviousStateWeighting.optimize()
        this.forgetInputWeighting.optimize()
        this.forgetBias?.optimize()

        this.shortTermMemoryWeighting.optimize()
        this.shortTermInputWeighting.optimize()
        this.shortTermBias?.optimize()

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

    val forgetActivations = Array(numberSteps) { indexStep ->

        val forgetActivationName = concatenateNames(name, "forget-activation-step-$indexStep")
        SigmoidLayer(forgetActivationName)

    }

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
        forgetPreviousStateWeighting,
        forgetInputWeighting,
        forgetAdditions,
        forgetBias,
        forgetActivations,
        shortTermForgetting,
        shortTermMemoryWeighting,
        shortTermInputWeighting,
        shortTermAdditions,
        shortTermBias,
        shortTermActivations,
        keepSubtractions,
        shortTermHadamards,
        longTermHadamards,
        stateAdditions)

    return minimalGatedUnit

}
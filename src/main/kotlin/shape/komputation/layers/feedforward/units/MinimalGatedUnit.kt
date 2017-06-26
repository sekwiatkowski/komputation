package shape.komputation.layers.feedforward.units

import shape.komputation.functions.add
import shape.komputation.layers.ContinuationLayer
import shape.komputation.layers.combination.AdditionCombination
import shape.komputation.layers.combination.HadamardCombination
import shape.komputation.layers.combination.SubtractionCombination
import shape.komputation.layers.feedforward.activation.SigmoidLayer
import shape.komputation.layers.feedforward.activation.TanhLayer
import shape.komputation.layers.feedforward.recurrent.SeriesBias
import shape.komputation.matrix.DoubleMatrix
import shape.komputation.matrix.doubleColumnVector
import shape.komputation.matrix.doubleOneColumnVector
import shape.komputation.matrix.doubleZeroColumnVector
import shape.komputation.optimization.DenseAccumulator

class MinimalGatedUnit(
    name : String?,
    hiddenDimension : Int,
    private val forgetInputProjection : ContinuationLayer,
    private val forgetPreviousStateProjection : ContinuationLayer,
    private val forgetAddition : AdditionCombination,
    private val forgetBias : SeriesBias?,
    private val forgetActivation : SigmoidLayer,
    private val shortTermInputProjection : ContinuationLayer,
    private val shortTermForgetting : HadamardCombination,
    private val shortTermMemoryProjection : ContinuationLayer,
    private val shortTermAddition : AdditionCombination,
    private val shortTermBias : SeriesBias?,
    private val shortTermActivation : TanhLayer,
    private val keepSubtraction : SubtractionCombination,
    private val longTermHadamard : HadamardCombination,
    private val shortTermHadamard: HadamardCombination,
    private val stateAddition: AdditionCombination) : RecurrentUnit(name) {

    private val one = doubleOneColumnVector(hiddenDimension)

    private val previousStateAccumulator = DenseAccumulator(hiddenDimension)
    private val inputAccumulator = DenseAccumulator(hiddenDimension)

    fun forwardForget(input : DoubleMatrix): DoubleMatrix {

        // forget projected input = forget input weights * input
        val forgetProjectedInput = this.forgetInputProjection.forward(input)

        // forget projected previous state = forget previous state weights * previous state
        val forgetProjectedPreviousInput = this.forgetPreviousStateProjection.forward(input)

        // forget addition = forget projected input + forget projected previous state
        val forgetAddition = this.forgetAddition.forward(forgetProjectedInput, forgetProjectedPreviousInput)

        // forget pre-activation = forget addition (+ forget bias)
        val forgetPreActivation =

            if (this.forgetBias == null) {

                forgetAddition

            }
            else {

                this.forgetBias.forwardStep(forgetAddition)
            }

        // forget = sigmoid(forget pre-activation)
        val forget = this.forgetActivation.forward(forgetPreActivation)

        return forget
    }

    fun forwardShortTermResponse(input : DoubleMatrix, forget : DoubleMatrix): DoubleMatrix {

        val shortTermMemory = this.shortTermForgetting.forward(input, forget)

        val shortTermProjectedMemory = this.shortTermMemoryProjection.forward(shortTermMemory)

        val shortTermProjectedInput = this.shortTermInputProjection.forward(input)

        val shortTermAddition = this.shortTermAddition.forward(shortTermProjectedMemory, shortTermProjectedInput)

        // forget pre-activation = forget addition (+ forget bias)
        val shortTermPreActivation =

            if (this.shortTermBias == null) {

                shortTermAddition

            }
            else {

                this.shortTermBias.forwardStep(shortTermAddition)
            }

        val shortTermResponse = this.shortTermActivation.forward(shortTermPreActivation)

        return shortTermResponse
    }

    override fun forward(state : DoubleMatrix, input : DoubleMatrix): DoubleMatrix {

        val forget = this.forwardForget(input)

        val keep = this.keepSubtraction.forward(one, forget)

        val longTermComponent = this.longTermHadamard.forward(keep, state)

        val shortTermResponse = this.forwardShortTermResponse(input, forget)

        val shortTermComponent = this.shortTermHadamard.forward(forget, shortTermResponse)

        val newState = this.stateAddition.forward(longTermComponent, shortTermComponent)

        return newState
    }

    fun backwardWrtForget(chain: DoubleMatrix) {

        // forget = sigmoid(forget pre-activation)

        // d forget / d forget pre-activation
        val diffForgetWrtPreActivation = this.forgetActivation.backward(chain)

        // forget pre-activation = forget addition (+ forget bias)
        // forget addition = forget projected input + forget projected previous state

        // d forget pre-activation / d forget addition = 1

        // d forget addition / d projected input
        val diffForgetPreActivationWrtProjectedInput = this.forgetAddition.backwardFirst(diffForgetWrtPreActivation)

        // d projected input / d input
        val diffForgetProjectedInputWrtInput = this.forgetInputProjection.backward(diffForgetPreActivationWrtProjectedInput)

        this.inputAccumulator.accumulate(diffForgetProjectedInputWrtInput.entries)

        // d forget addition / d projected previous state
        val diffForgetPreActivationWrtProjectedPreviousState = this.forgetAddition.backwardSecond(diffForgetWrtPreActivation)

        // d projected previous state / d previous state
        val diffForgetProjectedPreviousStateWrtPreviousState = this.forgetPreviousStateProjection.backward(diffForgetPreActivationWrtProjectedPreviousState)

        this.previousStateAccumulator.accumulate(diffForgetProjectedPreviousStateWrtPreviousState.entries)

        if (this.forgetBias != null) {

            // d forget addition / d projected input
            this.forgetBias.backwardStep(diffForgetWrtPreActivation)

        }

    }

    override fun backward(chain : DoubleMatrix): Pair<DoubleMatrix, DoubleMatrix> {

        // new state = long-term component + short-term component

        // d new state / d long-term component
        val diffChainWrtLongTermComponent = this.stateAddition.backwardFirst(chain)

        // d new state / d short-term component
        val diffChainWrtShortTermComponent = this.stateAddition.backwardSecond(chain)



        // long-term component = keep (.) previous state

        // d long-term component / d keep = previous state
        val diffLongTermComponentWrtKeep = this.longTermHadamard.backwardFirst(diffChainWrtLongTermComponent)

        // keep = 1 - forget

        // d (1 - forget) / d forget = -1
        val diffKeepWrtForget = this.keepSubtraction.backwardSecond(diffLongTermComponentWrtKeep)

        this.backwardWrtForget(diffKeepWrtForget)

        // d long-term component / d previous state = keep
        val diffLongTermComponentWrtPreviousState = this.longTermHadamard.backwardSecond(diffChainWrtLongTermComponent)

        this.previousStateAccumulator.accumulate(diffLongTermComponentWrtPreviousState.entries)



        // short-term component = forget (.) short-term response

        // d short-term component / forget = short-term response
        val diffShortTermComponentWrtForget = this.shortTermHadamard.backwardFirst(diffChainWrtShortTermComponent)

        this.backwardWrtForget(diffShortTermComponentWrtForget)

        // d short-term component / short-term response = forget
        val diffShortTermComponentWrtShortTermResponse = this.shortTermHadamard.backwardFirst(diffChainWrtShortTermComponent)

        // short-term response = tanh(short-term response pre-activation)
        // d short-term response / d short-term response pre-activation
        val diffShortTermResponseWrtPreActivation = this.shortTermActivation.backward(diffShortTermComponentWrtShortTermResponse)

        // short-term response pre-activation = projected input + projected short-term memory (+ short-term bias)

        // d short-term response pre-activation / d short-term projected input
        val diffShortTermResponsePreActivationWrtProjectedInput = this.shortTermAddition.backwardFirst(diffShortTermResponseWrtPreActivation)

        // d short-term projected input / d projected input
        val diffShortTermProjectedInputWrtInput = this.shortTermInputProjection.backward(diffShortTermResponsePreActivationWrtProjectedInput)

        this.inputAccumulator.accumulate(diffShortTermProjectedInputWrtInput.entries)

        // d short-term response pre-activation / d projected short-term memory
        val diffShortTermResponsePreActivationWrtProjectedShortTermMemory = this.shortTermAddition.backwardSecond(diffShortTermResponseWrtPreActivation)

        // d projected short-term memory / d short-term memory
        val diffProjectedShortTermMemoryWrtShortTermMemory = this.shortTermMemoryProjection.backward(diffShortTermResponsePreActivationWrtProjectedShortTermMemory)

        // short-term memory = forget (.) previous state
        val diffShortTermMemoryWrtForget = this.shortTermForgetting.backwardFirst(diffProjectedShortTermMemoryWrtShortTermMemory)

        this.backwardWrtForget(diffShortTermMemoryWrtForget)



        // short-term memory = forget (.) previous state
        val diffShortTermMemoryWrtPreviousState = this.shortTermForgetting.backwardFirst(diffProjectedShortTermMemoryWrtShortTermMemory)

        this.previousStateAccumulator.accumulate(diffShortTermMemoryWrtPreviousState.entries)

        if (this.shortTermBias != null) {

            this.shortTermBias.backwardStep(diffShortTermResponseWrtPreActivation)

        }

        return doubleColumnVector(*this.previousStateAccumulator.getAccumulation()) to doubleColumnVector(*this.inputAccumulator.getAccumulation())

    }

}


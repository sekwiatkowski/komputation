package shape.komputation.layers.forward.units

import shape.komputation.layers.combination.AdditionCombination
import shape.komputation.layers.combination.HadamardCombination
import shape.komputation.layers.forward.activation.TanhLayer
import shape.komputation.layers.forward.projection.SeriesBias
import shape.komputation.layers.forward.projection.SeriesWeighting
import shape.komputation.matrix.DoubleMatrix


class ShortTermResponse(
    private val forgetting: Array<HadamardCombination>,
    private val memoryWeighting: SeriesWeighting,
    private val inputWeighting: SeriesWeighting,
    private val additions: Array<AdditionCombination>,
    private val bias: SeriesBias?,
    private val activations: Array<TanhLayer>) {

    fun forward(step : Int, state : DoubleMatrix, input : DoubleMatrix, forget : DoubleMatrix, isTraining : Boolean): DoubleMatrix {

        val shortTermMemory = this.forgetting[step].forward(state, forget)

        val weightedShortTermMemory = this.memoryWeighting.forwardStep(step, shortTermMemory, isTraining)

        val weightedInput = this.inputWeighting.forwardStep(step, input, isTraining)

        val addition = this.additions[step].forward(weightedShortTermMemory, weightedInput)

        val preActivation =

            if (this.bias == null) {

                addition

            }
            else {

                this.bias.forwardStep(addition)
            }

        val shortTermResponse = this.activations[step].forward(preActivation, isTraining)

        return shortTermResponse

    }

    fun backward(step: Int, chain: DoubleMatrix): Pair<DoubleMatrix, Pair<DoubleMatrix, DoubleMatrix>> {

        // short-term response = tanh(short-term response pre-activation)
        // d short-term response / d short-term response pre-activation
        val diffShortTermResponseWrtPreActivation = this.activations[step].backward(chain)

        // short-term response pre-activation = weighted short-term memory + weighted input (+ short-term bias)

        // d short-term response pre-activation / d weighted short-term memory
        val diffPreActivationWrtWeightedShortTermMemory = this.additions[step].backwardFirst(diffShortTermResponseWrtPreActivation)

        // d weighted short-term memory / d short-term memory
        val diffWeightedShortTermMemoryWrtShortTermMemory = this.memoryWeighting.backwardStep(step, diffPreActivationWrtWeightedShortTermMemory)

        // d short-term memory / d forget
        val diffShortTermMemoryWrtForget = this.forgetting[step].backwardFirst(diffWeightedShortTermMemoryWrtShortTermMemory)

        // d short-term memory / d previous state
        val diffShortTermMemoryWrtPreviousState = this.forgetting[step].backwardFirst(diffWeightedShortTermMemoryWrtShortTermMemory)

        // d short-term response pre-activation / d short-term weighted input
        val diffPreActivationWrtWeightedInput = this.additions[step].backwardSecond(diffShortTermResponseWrtPreActivation)

        // d short-term weighted input / d weighted input
        val diffWeightedInputWrtInput = this.inputWeighting.backwardStep(step, diffPreActivationWrtWeightedInput)

        if (this.bias != null) {

            bias.backwardStep(diffShortTermResponseWrtPreActivation)

        }

        return diffShortTermMemoryWrtForget to (diffShortTermMemoryWrtPreviousState to diffWeightedInputWrtInput)

    }

    fun backwardSeries() {

        this.memoryWeighting.backwardSeries()
        this.inputWeighting.backwardSeries()
        this.bias?.backwardSeries()

    }

    fun optimize() {

        this.memoryWeighting.optimize()
        this.inputWeighting.optimize()
        this.bias?.optimize()

    }

}
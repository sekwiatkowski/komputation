package shape.komputation.cpu.forward.units

import shape.komputation.cpu.combination.AdditionCombination
import shape.komputation.cpu.combination.HadamardCombination
import shape.komputation.cpu.combination.hadamardCombination
import shape.komputation.cpu.forward.activation.CpuTanhLayer
import shape.komputation.cpu.forward.projection.SeriesBias
import shape.komputation.cpu.forward.projection.SeriesWeighting
import shape.komputation.cpu.forward.projection.seriesBias
import shape.komputation.cpu.forward.projection.seriesWeighting
import shape.komputation.initialization.InitializationStrategy
import shape.komputation.layers.concatenateNames
import shape.komputation.layers.forward.activation.tanhLayer
import shape.komputation.matrix.DoubleMatrix
import shape.komputation.optimization.OptimizationStrategy

class ShortTermResponse(
    private val forgetting: Array<HadamardCombination>,
    private val memoryWeighting: SeriesWeighting,
    private val inputWeighting: SeriesWeighting,
    private val additions: Array<AdditionCombination>,
    private val bias: SeriesBias?,
    private val activations: Array<CpuTanhLayer>) {

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

    fun optimize(scalingFactor : Double) {

        this.memoryWeighting.optimize(scalingFactor)
        this.inputWeighting.optimize(scalingFactor)
        this.bias?.optimize(scalingFactor)

    }

}

fun shortTermResponse(
    name : String,
    numberSteps : Int,
    hiddenDimension : Int,
    inputDimension : Int,
    memoryWeightInitializationStrategy : InitializationStrategy,
    inputWeightInitializationStrategy : InitializationStrategy,
    biasInitializationStrategy: InitializationStrategy?,
    optimizationStrategy : OptimizationStrategy?): ShortTermResponse {

    val shortTermForgetting = Array(numberSteps) { indexStep ->

        val shortTermForgettingName = concatenateNames(name, "forgetting-step-$indexStep")
        hadamardCombination(shortTermForgettingName)

    }

    val shortTermMemoryWeightingSeriesName = concatenateNames(name, "memory-weighting")
    val shortTermMemoryWeightingStepName = concatenateNames(name, "memory-weighting-step")
    val shortTermMemoryWeighting = seriesWeighting(shortTermMemoryWeightingSeriesName, shortTermMemoryWeightingStepName, numberSteps, true, hiddenDimension, hiddenDimension, memoryWeightInitializationStrategy, optimizationStrategy)

    val shortTermInputWeightingSeriesName = concatenateNames(name, "input-weighting")
    val shortTermInputWeightingStepName = concatenateNames(name, "input-weighting-step")
    val shortTermInputWeighting = seriesWeighting(shortTermInputWeightingSeriesName, shortTermInputWeightingStepName, numberSteps, false, inputDimension, hiddenDimension, inputWeightInitializationStrategy, optimizationStrategy)

    val shortTermAdditions = Array(numberSteps) { indexStep ->

        val shortTermAdditionName = concatenateNames(name, "addition-step-$indexStep")
        AdditionCombination(shortTermAdditionName)

    }

    val shortTermBias =

        if (biasInitializationStrategy != null) {

            val shortTermBiasName = concatenateNames(name, "bias")
            seriesBias(shortTermBiasName, hiddenDimension, biasInitializationStrategy, optimizationStrategy)

        }
        else {
            null
        }


    val shortTermActivations = Array(numberSteps) { indexStep ->

        val shortTermActivationName = concatenateNames(name, "activation-step-$indexStep")
        tanhLayer(shortTermActivationName).buildForCpu()

    }

    return ShortTermResponse(shortTermForgetting, shortTermMemoryWeighting, shortTermInputWeighting, shortTermAdditions, shortTermBias, shortTermActivations)

}
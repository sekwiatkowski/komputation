package shape.komputation.layers.feedforward.decoder

import shape.komputation.functions.extractStep
import shape.komputation.layers.ContinuationLayer
import shape.komputation.layers.OptimizableLayer
import shape.komputation.matrix.DoubleMatrix
import shape.komputation.matrix.SequenceMatrix
import shape.komputation.matrix.doubleZeroColumnVector
import shape.komputation.matrix.zeroSequenceMatrix

class MultiInputDecoder(
    name : String?,
    private val unit : DecoderUnit,
    private val numberSteps : Int,
    private val inputDimension: Int,
    private val hiddenDimension: Int,
    private val outputDimension : Int) : ContinuationLayer(name), OptimizableLayer {

    override fun forward(input: DoubleMatrix): DoubleMatrix {

        input as SequenceMatrix

        val seriesOutput = zeroSequenceMatrix(this.numberSteps, this.outputDimension)

        // Start with a zero state
        var state = doubleZeroColumnVector(this.hiddenDimension)

        for (indexStep in 0..this.numberSteps - 1) {

            // Extract the n-th step input
            val stepInput = input.getStep(indexStep)

            // Forward the current state together the current step input
            val (newState, newOutput) = this.unit.forward(indexStep, state, stepInput)

            // Store the n-th output
            seriesOutput.setStep(indexStep, newOutput.entries)

            state = newState

        }

        return seriesOutput

    }

    // Incoming gradient: d chain / d series prediction
    override fun backward(chain: DoubleMatrix): DoubleMatrix {

        val chainEntries = chain.entries

        val backwardStatePreActivationWrtInput = zeroSequenceMatrix(this.numberSteps, this.inputDimension)
        var backwardStatePreActivationWrtPreviousState : DoubleMatrix? = null

        for (indexStep in this.numberSteps - 1 downTo 0) {

            val isLastStep = indexStep + 1 == this.numberSteps

            val chainStep = extractStep(chainEntries, indexStep, outputDimension)

            val (newBackwardStatePreActivationWrtInput, newBackwardStatePreActivationWrtPreviousState) = this.unit.backwardStep(isLastStep, indexStep, chainStep, backwardStatePreActivationWrtInput, backwardStatePreActivationWrtPreviousState)

            backwardStatePreActivationWrtInput.setStep(indexStep, newBackwardStatePreActivationWrtInput.entries)
            backwardStatePreActivationWrtPreviousState = newBackwardStatePreActivationWrtPreviousState

        }

        this.unit.backwardSeries()

        return backwardStatePreActivationWrtInput

    }

    override fun optimize() {

        this.unit.optimize()

    }

}

fun createMultiInputDecoder(
    unit : DecoderUnit,
    numberSteps: Int,
    inputDimension: Int,
    hiddenDimension: Int,
    outputDimension: Int) =

    MultiInputDecoder(
        null,
        unit,
        numberSteps,
        inputDimension,
        hiddenDimension,
        outputDimension)

fun createMultiInputDecoder(
    name : String?,
    unit : DecoderUnit,
    numberSteps: Int,
    inputDimension: Int,
    hiddenDimension: Int,
    outputDimension: Int) =

    MultiInputDecoder(
        name,
        unit,
        numberSteps,
        inputDimension,
        hiddenDimension,
        outputDimension)
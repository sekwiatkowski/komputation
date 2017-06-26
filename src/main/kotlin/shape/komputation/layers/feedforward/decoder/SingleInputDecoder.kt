package shape.komputation.layers.feedforward.decoder

import shape.komputation.functions.add
import shape.komputation.functions.extractStep
import shape.komputation.layers.ContinuationLayer
import shape.komputation.layers.OptimizableLayer
import shape.komputation.matrix.DoubleMatrix
import shape.komputation.matrix.doubleColumnVector
import shape.komputation.matrix.doubleZeroColumnVector
import shape.komputation.matrix.zeroSequenceMatrix


// The first input is empty.
// Starting with the second input, the input at step t is the output of step t-1.
class SingleInputDecoder(
    name : String?,
    private val unit : DecoderUnit,
    private val numberSteps : Int,
    private val outputDimension : Int) : ContinuationLayer(name), OptimizableLayer {

    override fun forward(encoderOutput: DoubleMatrix): DoubleMatrix {

        val seriesOutput = zeroSequenceMatrix(this.numberSteps, this.outputDimension)

        // Use the encoder output as the first state
        var state = encoderOutput
        // The first input is empty since there is no previous output
        var input = doubleZeroColumnVector(outputDimension)

        for (indexStep in 0..this.numberSteps - 1) {

            val (newState, newOutput) = this.unit.forward(indexStep, state, input)

            seriesOutput.setStep(indexStep, newOutput.entries)

            state = newState
            // The output of step t-1 is the input for step t.
            input = newOutput

        }

        return seriesOutput

    }

    // Incoming gradient: d chain / d series prediction
    override fun backward(chain: DoubleMatrix): DoubleMatrix {

        val chainEntries = chain.entries

        // Differentiate the chain w.r.t. input
        var backwardStatePreActivationWrtInput : DoubleMatrix? = null

        // Differentiate the chain w.r.t previous state.
        // This is done at each step. For the first step (t=1), the chain is differentiated w.r.t. to the initial state (t=0).
        var backwardStatePreActivationWrtPreviousState : DoubleMatrix? = null

        for (indexStep in this.numberSteps - 1 downTo 0) {

            val chainStep = extractStep(chainEntries, indexStep, outputDimension)

            val outputSum = doubleColumnVector(*

                // The input gradient for step t+1 is added to the chain step t ...
                if (backwardStatePreActivationWrtInput != null) {

                    // d chain / d output(index+1) * d output(index+1) / d input(index + 1) *  d input(index + 1) / d output(index)
                    add(chainStep, backwardStatePreActivationWrtInput.entries)

                }
                // ... except in the case of the last step (t = T)
                else {

                    chainStep

                })

            val (newBackwardStatePreActivationWrtInput, newBackwardStatePreActivationWrtPreviousState) = this.unit.backwardStep(indexStep, outputSum, backwardStatePreActivationWrtPreviousState)

            backwardStatePreActivationWrtInput = newBackwardStatePreActivationWrtInput
            backwardStatePreActivationWrtPreviousState = newBackwardStatePreActivationWrtPreviousState

        }

        this.unit.backwardSeries()

        return backwardStatePreActivationWrtPreviousState!!

    }

    override fun optimize() {

        this.unit.optimize()

    }

}

fun createSingleInputDecoder(
    unit : DecoderUnit,
    numberSteps: Int,
    outputDimension: Int) =

    createSingleInputDecoder(
        null,
        unit,
        numberSteps,
        outputDimension)


fun createSingleInputDecoder(
    name : String?,
    unit : DecoderUnit,
    numberSteps: Int,
    outputDimension: Int) =

    SingleInputDecoder(
        name,
        unit,
        numberSteps,
        outputDimension)
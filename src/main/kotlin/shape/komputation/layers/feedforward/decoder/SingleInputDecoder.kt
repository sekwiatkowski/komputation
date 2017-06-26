package shape.komputation.layers.feedforward.decoder

import shape.komputation.functions.extractStep
import shape.komputation.layers.ContinuationLayer
import shape.komputation.layers.OptimizableLayer
import shape.komputation.matrix.DoubleMatrix
import shape.komputation.matrix.doubleZeroColumnVector
import shape.komputation.matrix.zeroSequenceMatrix

class SingleInputDecoder(
    name : String?,
    private val unit : DecoderUnit,
    private val numberSteps : Int,
    private val outputDimension : Int) : ContinuationLayer(name), OptimizableLayer {

    override fun forward(input: DoubleMatrix): DoubleMatrix {

        val seriesOutput = zeroSequenceMatrix(this.numberSteps, this.outputDimension)

        var state = input
        var previousOutput = doubleZeroColumnVector(outputDimension)

        for (indexStep in 0..this.numberSteps - 1) {

            val (newState, newOutput) = this.unit.forward(indexStep, state, previousOutput)

            seriesOutput.setStep(indexStep, newOutput.entries)

            state = newState
            previousOutput = newOutput

        }

        return seriesOutput

    }

    // Incoming gradient: d chain / d series prediction
    override fun backward(chain: DoubleMatrix): DoubleMatrix {

        val chainEntries = chain.entries

        var backwardStatePreActivationWrtInput : DoubleMatrix? = null
        var backwardStatePreActivationWrtPreviousState : DoubleMatrix? = null

        for (indexStep in this.numberSteps - 1 downTo 0) {

            val stepChain = extractStep(chainEntries, indexStep, outputDimension)

            val isLastStep = indexStep + 1 == this.numberSteps

            val (newBackwardStatePreActivationWrtInput, newBackwardStatePreActivationWrtPreviousState) = this.unit.backwardStep(isLastStep, indexStep, stepChain, backwardStatePreActivationWrtInput, backwardStatePreActivationWrtPreviousState)

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
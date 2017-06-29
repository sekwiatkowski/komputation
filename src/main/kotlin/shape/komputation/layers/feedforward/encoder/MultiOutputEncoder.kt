package shape.komputation.layers.feedforward.encoder

import shape.komputation.functions.add
import shape.komputation.functions.extractStep
import shape.komputation.layers.ContinuationLayer
import shape.komputation.layers.feedforward.units.RecurrentUnit
import shape.komputation.matrix.*
import shape.komputation.optimization.Optimizable

class MultiOutputEncoder(
    name : String?,
    isReversed : Boolean,
    private val unit: RecurrentUnit,
    private val numberSteps: Int,
    private val inputDimension: Int,
    private val hiddenDimension : Int) : ContinuationLayer(name), Optimizable {

    private val startAtTheBeginning = 0..numberSteps - 1
    private val startAtTheEnd = this.numberSteps - 1 downTo 0

    private val stepIndices = if(isReversed) IntArray(this.numberSteps) { index -> this.numberSteps - 1 - index } else IntArray(this.numberSteps) { index -> index }

    override fun forward(input: DoubleMatrix): DoubleMatrix {

        var state = doubleZeroColumnVector(hiddenDimension)

        input as SequenceMatrix

        val output = zeroSequenceMatrix(numberSteps, hiddenDimension, 1)

        for (indexStep in this.startAtTheBeginning) {

            val stepInput = input.getStep(this.stepIndices[indexStep])

            state = this.unit.forwardStep(indexStep, state, stepInput)

            output.setStep(indexStep, state.entries)


        }

        return output

    }

    override fun backward(incoming: DoubleMatrix): DoubleMatrix {

        val seriesBackwardWrtInput = zeroSequenceMatrix(this.numberSteps, inputDimension)

        var stateChain = EMPTY_DOUBLE_MATRIX

        val incomingEntries = incoming.entries

        for (indexStep in this.startAtTheEnd) {

            val isLastStep = indexStep + 1 == this.numberSteps

            val incomingStepEntries = extractStep(incomingEntries, indexStep, hiddenDimension)

            val chainEntries =

                if (isLastStep) {

                    incomingStepEntries
                }
                else {

                    add(stateChain.entries, incomingStepEntries)

                }

            val (backwardStatePreActivationWrtPreviousState, backwardStatePreActivationWrtInput) = this.unit.backwardStep(indexStep, doubleColumnVector(*chainEntries))

            stateChain = backwardStatePreActivationWrtPreviousState

            seriesBackwardWrtInput.setStep(this.stepIndices[indexStep], backwardStatePreActivationWrtInput.entries)

        }

        this.unit.backwardSeries()

        return stateChain

    }

    override fun optimize() {

        if (this.unit is Optimizable) {

            this.unit.optimize()

        }

    }

}

fun createMultiOutputEncoder(
    unit : RecurrentUnit,
    numberSteps : Int,
    inputDimension : Int,
    hiddenDimension: Int,
    isReversed : Boolean = false) =

    createMultiOutputEncoder(
        null,
        unit,
        numberSteps,
        inputDimension,
        hiddenDimension,
        isReversed
    )

fun createMultiOutputEncoder(
    name : String?,
    unit : RecurrentUnit,
    numberSteps : Int,
    inputDimension : Int,
    hiddenDimension : Int,
    isReversed : Boolean = false) =

    MultiOutputEncoder(
        name,
        isReversed,
        unit,
        numberSteps,
        inputDimension,
        hiddenDimension
    )
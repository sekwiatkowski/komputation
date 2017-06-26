package shape.komputation.layers.feedforward.encoder

import shape.komputation.functions.add
import shape.komputation.functions.extractStep
import shape.komputation.layers.ContinuationLayer
import shape.komputation.layers.OptimizableLayer
import shape.komputation.layers.feedforward.units.RecurrentUnit
import shape.komputation.matrix.*

class MultiOutputEncoder(
    name : String?,
    private val unit: RecurrentUnit,
    private val numberSteps: Int,
    private val inputDimension: Int,
    private val hiddenDimension : Int) : ContinuationLayer(name), OptimizableLayer {

    override fun forward(input: DoubleMatrix): DoubleMatrix {

        var state = doubleZeroColumnVector(hiddenDimension)

        input as SequenceMatrix

        val output = zeroSequenceMatrix(numberSteps, hiddenDimension, 1)

        for (indexStep in 0..numberSteps - 1) {

            val stepInput = input.getStep(indexStep)

            state = this.unit.forwardStep(indexStep, state, stepInput)

            output.setStep(indexStep, state.entries)


        }

        return output

    }

    override fun backward(incoming: DoubleMatrix): DoubleMatrix {

        val seriesBackwardWrtInput = zeroSequenceMatrix(this.numberSteps, inputDimension)

        var stateChain = EMPTY_DOUBLE_MATRIX

        val incomingEntries = incoming.entries

        for (indexStep in this.numberSteps - 1 downTo 0) {

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

            seriesBackwardWrtInput.setStep(indexStep, backwardStatePreActivationWrtInput.entries)

        }

        this.unit.backwardSeries()

        return stateChain

    }

    override fun optimize() {

        if (this.unit is OptimizableLayer) {

            this.unit.optimize()

        }

    }

}

fun createMultiOutputEncoder(
    unit : RecurrentUnit,
    numberSteps : Int,
    inputDimension : Int,
    hiddenDimension: Int) =

    createMultiOutputEncoder(
        null,
        unit,
        numberSteps,
        inputDimension,
        hiddenDimension
    )

fun createMultiOutputEncoder(
    name : String?,
    unit : RecurrentUnit,
    numberSteps : Int,
    inputDimension : Int,
    hiddenDimension: Int) =

    MultiOutputEncoder(
        name,
        unit,
        numberSteps,
        inputDimension,
        hiddenDimension
    )
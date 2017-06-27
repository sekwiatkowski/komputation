package shape.komputation.layers.feedforward.encoder

import shape.komputation.layers.ContinuationLayer
import shape.komputation.layers.OptimizableLayer
import shape.komputation.layers.feedforward.units.RecurrentUnit
import shape.komputation.matrix.*

class SingleOutputEncoder(
    name : String?,
    private val unit: RecurrentUnit,
    private val numberSteps: Int,
    private val inputDimension: Int,
    private val hiddenDimension : Int) : ContinuationLayer(name), OptimizableLayer {

    override fun forward(input: DoubleMatrix): DoubleMatrix {

        input as SequenceMatrix

        var currentState = doubleZeroColumnVector(hiddenDimension)

        for (indexStep in 0..numberSteps - 1) {

            val stepInput = input.getStep(indexStep)

            currentState = this.unit.forwardStep(indexStep, currentState, stepInput)

        }

        return currentState

    }

    override fun backward(incoming: DoubleMatrix): DoubleMatrix {

        val seriesBackwardWrtInput = zeroSequenceMatrix(this.numberSteps, this.inputDimension)

        var stateChain = incoming

        for (indexStep in this.numberSteps - 1 downTo 0) {

            val (backwardWrtPreviousState, backwardWrtInput) = this.unit.backwardStep(indexStep, stateChain)

            stateChain = backwardWrtPreviousState

            seriesBackwardWrtInput.setStep(indexStep, backwardWrtInput.entries)

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

fun createSingleOutputEncoder(
    unit : RecurrentUnit,
    numberSteps : Int,
    inputDimension : Int,
    hiddenDimension: Int) =

    createSingleOutputEncoder(
        null,
        unit,
        numberSteps,
        inputDimension,
        hiddenDimension
    )

fun createSingleOutputEncoder(
    name : String?,
    unit : RecurrentUnit,
    numberSteps : Int,
    inputDimension : Int,
    hiddenDimension: Int) =

    SingleOutputEncoder(
        name,
        unit,
        numberSteps,
        inputDimension,
        hiddenDimension
    )
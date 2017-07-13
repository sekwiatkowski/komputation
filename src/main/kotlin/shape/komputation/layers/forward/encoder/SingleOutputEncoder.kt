package shape.komputation.layers.forward.encoder

import shape.komputation.layers.BaseForwardLayer
import shape.komputation.layers.CpuForwardLayerInstruction
import shape.komputation.layers.forward.units.RecurrentUnit
import shape.komputation.matrix.DoubleMatrix
import shape.komputation.matrix.doubleZeroColumnVector
import shape.komputation.matrix.doubleZeroMatrix
import shape.komputation.optimization.Optimizable

class CpuSingleOutputEncoder internal constructor(
    name : String?,
    isReversed : Boolean,
    private val unit: RecurrentUnit,
    private val numberSteps: Int,
    private val inputDimension: Int,
    private val hiddenDimension : Int) : BaseForwardLayer(name), Optimizable {

    private val startAtTheBeginning = 0..numberSteps - 1
    private val startAtTheEnd = this.numberSteps - 1 downTo 0

    private val stepIndices = if(isReversed) IntArray(this.numberSteps) { index -> this.numberSteps - 1 - index } else IntArray(this.numberSteps) { index -> index }

    override fun forward(input: DoubleMatrix, isTraining : Boolean): DoubleMatrix {

        var currentState = doubleZeroColumnVector(this.hiddenDimension)

        for (indexStep in this.startAtTheBeginning) {

            val stepInput = input.getColumn(this.stepIndices[indexStep])

            currentState = this.unit.forwardStep(indexStep, currentState, stepInput, isTraining)

        }

        return currentState

    }

    override fun backward(incoming: DoubleMatrix): DoubleMatrix {

        val seriesBackwardWrtInput = doubleZeroMatrix(this.inputDimension, this.numberSteps)

        var stateChain = incoming

        for (indexStep in this.startAtTheEnd) {

            val (diffWrtPreviousState, diffWrtInput) = this.unit.backwardStep(indexStep, stateChain)

            stateChain = diffWrtPreviousState

            seriesBackwardWrtInput.setColumn(this.stepIndices[indexStep], diffWrtInput.entries)

        }

        this.unit.backwardSeries()

        return stateChain

    }

    override fun optimize(scalingFactor : Double) {

        if (this.unit is Optimizable) {

            this.unit.optimize(scalingFactor)

        }

    }

}

class SingleOutputEncoder(
    private val name : String?,
    private val unit : RecurrentUnit,
    private val numberSteps : Int,
    private val inputDimension : Int,
    private val hiddenDimension: Int,
    private val isReversed: Boolean) : CpuForwardLayerInstruction {

    override fun buildForCpu() =

        CpuSingleOutputEncoder(
            this.name,
            this.isReversed,
            this.unit,
            this.numberSteps,
            this.inputDimension,
            this.hiddenDimension
        )


}

fun singleOutputEncoder(
    unit : RecurrentUnit,
    numberSteps : Int,
    inputDimension : Int,
    hiddenDimension: Int,
    isReversed: Boolean = false) =

    singleOutputEncoder(
        null,
        unit,
        numberSteps,
        inputDimension,
        hiddenDimension,
        isReversed
    )

fun singleOutputEncoder(
    name : String?,
    unit : RecurrentUnit,
    numberSteps : Int,
    inputDimension : Int,
    hiddenDimension: Int,
    isReversed: Boolean = false) =

    SingleOutputEncoder(name, unit, numberSteps, inputDimension, hiddenDimension, isReversed)
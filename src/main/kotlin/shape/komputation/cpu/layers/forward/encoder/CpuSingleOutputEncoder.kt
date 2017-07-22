package shape.komputation.cpu.layers.forward.encoder

import shape.komputation.cpu.layers.BaseCpuForwardLayer
import shape.komputation.cpu.layers.forward.units.RecurrentUnit
import shape.komputation.matrix.FloatMatrix
import shape.komputation.matrix.floatZeroColumnVector
import shape.komputation.matrix.floatZeroMatrix
import shape.komputation.optimization.Optimizable

class CpuSingleOutputEncoder internal constructor(
    name : String?,
    isReversed : Boolean,
    private val unit: RecurrentUnit,
    private val numberSteps: Int,
    private val inputDimension: Int,
    private val hiddenDimension : Int) : BaseCpuForwardLayer(name), Optimizable {

    private val startAtTheBeginning = 0..numberSteps - 1
    private val startAtTheEnd = this.numberSteps - 1 downTo 0

    private val stepIndices = if(isReversed) IntArray(this.numberSteps) { index -> this.numberSteps - 1 - index } else IntArray(this.numberSteps) { index -> index }

    override fun forward(input: FloatMatrix, isTraining : Boolean): FloatMatrix {

        var currentState = floatZeroColumnVector(this.hiddenDimension)

        for (indexStep in this.startAtTheBeginning) {

            val stepInput = input.getColumn(this.stepIndices[indexStep])

            currentState = this.unit.forwardStep(indexStep, currentState, stepInput, isTraining)

        }

        return currentState

    }

    override fun backward(incoming: FloatMatrix): FloatMatrix {

        val seriesBackwardWrtInput = floatZeroMatrix(this.inputDimension, this.numberSteps)

        var stateChain = incoming

        for (indexStep in this.startAtTheEnd) {

            val (diffWrtPreviousState, diffWrtInput) = this.unit.backwardStep(indexStep, stateChain)

            stateChain = diffWrtPreviousState

            seriesBackwardWrtInput.setColumn(this.stepIndices[indexStep], diffWrtInput.entries)

        }

        this.unit.backwardSeries()

        return stateChain

    }

    override fun optimize(scalingFactor : Float) {

        if (this.unit is Optimizable) {

            this.unit.optimize(scalingFactor)

        }

    }

}
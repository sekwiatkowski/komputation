package shape.komputation.cpu.layers.forward.encoder

import shape.komputation.cpu.functions.getStep
import shape.komputation.cpu.functions.setStep
import shape.komputation.cpu.layers.BaseCpuForwardLayer
import shape.komputation.cpu.layers.forward.units.RecurrentUnit
import shape.komputation.matrix.FloatMatrix
import shape.komputation.matrix.floatZeroColumnVector
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

    private val inputStepEntries = FloatArray(this.inputDimension)
    private val inputStep = FloatMatrix(this.inputDimension, 1, this.inputStepEntries)

    override fun forward(withinBatch : Int, input: FloatMatrix, isTraining : Boolean): FloatMatrix {

        var currentState = floatZeroColumnVector(this.hiddenDimension)

        val inputEntries = input.entries

        for (indexStep in this.startAtTheBeginning) {

            getStep(inputEntries, this.stepIndices[indexStep], this.inputStepEntries, this.inputDimension)

            currentState = this.unit.forwardStep(withinBatch, indexStep, currentState, this.inputStep, isTraining)

        }

        return currentState

    }

    override fun backward(withinBatch: Int, incoming: FloatMatrix): FloatMatrix {

        val seriesBackwardWrtInput = FloatArray(this.inputDimension * this.numberSteps)

        var stateChain = incoming

        for (indexStep in this.startAtTheEnd) {

            val (diffWrtPreviousState, diffWrtInput) = this.unit.backwardStep(withinBatch, indexStep, stateChain)

            stateChain = diffWrtPreviousState

            setStep(seriesBackwardWrtInput, this.stepIndices[indexStep], diffWrtInput.entries, this.inputDimension)

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
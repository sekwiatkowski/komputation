package shape.komputation.cpu.layers.forward.encoder

import shape.komputation.cpu.functions.add
import shape.komputation.cpu.functions.getStep
import shape.komputation.cpu.functions.setStep
import shape.komputation.cpu.layers.BaseCpuForwardLayer
import shape.komputation.cpu.layers.forward.units.RecurrentUnit
import shape.komputation.matrix.FloatMatrix
import shape.komputation.matrix.floatColumnVector
import shape.komputation.matrix.floatZeroColumnVector
import shape.komputation.matrix.floatZeroMatrix
import shape.komputation.optimization.Optimizable

class CpuMultiOutputEncoder internal constructor(
    name : String?,
    isReversed : Boolean,
    private val unit: RecurrentUnit,
    private val numberSteps: Int,
    private val inputDimension: Int,
    private val hiddenDimension : Int) : BaseCpuForwardLayer(name), Optimizable {

    private val startAtTheBeginning = 0..numberSteps - 1
    private val startAtTheEnd = this.numberSteps - 1 downTo 0

    private val stepIndices = if(isReversed) IntArray(this.numberSteps) { index -> this.numberSteps - 1 - index } else IntArray(this.numberSteps) { index -> index }

    private val forwardEntries = FloatArray(this.numberSteps * this.hiddenDimension)

    private val inputStepEntries = FloatArray(this.inputDimension)
    private val inputStep = FloatMatrix(this.inputDimension, 1, inputStepEntries)

    override fun forward(withinBatch : Int, input: FloatMatrix, isTraining : Boolean): FloatMatrix {

        var state = floatZeroColumnVector(this.hiddenDimension)

        val inputEntries = input.entries

        for (indexStep in this.startAtTheBeginning) {

            getStep(inputEntries, this.stepIndices[indexStep], this.inputStepEntries, this.inputDimension)

            state = this.unit.forwardStep(withinBatch, indexStep, state, this.inputStep, isTraining)

            setStep(this.forwardEntries, indexStep, state.entries, this.hiddenDimension)

        }

        return FloatMatrix(this.hiddenDimension, this.numberSteps, this.forwardEntries)

    }

    override fun backward(withinBatch : Int, incoming: FloatMatrix): FloatMatrix {

        val seriesBackwardWrtInput = floatZeroMatrix(this.inputDimension, this.numberSteps)

        var stateChain = floatZeroColumnVector(this.hiddenDimension)

        val incomingEntries = incoming.entries

        val incomingStepEntries = FloatArray(this.hiddenDimension)

        for (indexStep in this.startAtTheEnd) {

            getStep(incomingEntries, indexStep, incomingStepEntries, this.hiddenDimension)

            val chainEntries =

                if (indexStep + 1 == this.numberSteps) {

                    incomingStepEntries
                }
                else {

                    add(stateChain.entries, incomingStepEntries, incomingStepEntries, this.hiddenDimension)

                    incomingStepEntries

                }

            val (backwardStatePreActivationWrtPreviousState, backwardStatePreActivationWrtInput) = this.unit.backwardStep(withinBatch, indexStep, floatColumnVector(*chainEntries))

            stateChain = backwardStatePreActivationWrtPreviousState

            setStep(seriesBackwardWrtInput.entries, indexStep, backwardStatePreActivationWrtInput.entries, this.inputDimension)

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
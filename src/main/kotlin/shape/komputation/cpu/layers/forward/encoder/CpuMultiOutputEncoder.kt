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
import java.util.*

class CpuMultiOutputEncoder internal constructor(
    name : String?,
    isReversed : Boolean,
    private val unit: RecurrentUnit,
    private val numberSteps: Int,
    private val inputDimension: Int,
    private val hiddenDimension : Int) : BaseCpuForwardLayer(name), Optimizable {

    private val startAtTheBeginning = 0..numberSteps - 1
    private val startAtTheEnd = this.numberSteps - 1 downTo 0

    private val inputIndices = if(isReversed) IntArray(this.numberSteps) { index -> this.numberSteps - 1 - index } else IntArray(this.numberSteps) { index -> index }
    private val steps = Array(this.numberSteps) { FloatArray(this.inputDimension) }
    private val states = Array(this.numberSteps+1) { FloatArray(this.hiddenDimension) }
    private val forwardEntries = FloatArray(this.numberSteps * this.hiddenDimension)

    override fun forward(withinBatch : Int, input: FloatMatrix, isTraining : Boolean): FloatMatrix {

        val inputEntries = input.entries

        for (indexStep in this.startAtTheBeginning) {

            val stepEntries = this.steps[indexStep]
            getStep(inputEntries, this.inputIndices[indexStep], stepEntries, this.inputDimension)

            val newState = this.unit.forwardStep(withinBatch, indexStep, floatColumnVector(*this.states[indexStep]), floatColumnVector(*stepEntries), isTraining).entries

            this.states[indexStep+1] = newState

            System.arraycopy(newState, 0, this.forwardEntries, indexStep * this.hiddenDimension, this.hiddenDimension)

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

            setStep(backwardStatePreActivationWrtInput.entries, indexStep, seriesBackwardWrtInput.entries, this.inputDimension)

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
package shape.komputation.layers.feedforward.encoder

import shape.komputation.functions.add
import shape.komputation.layers.ContinuationLayer
import shape.komputation.layers.feedforward.recurrent.SeriesBias
import shape.komputation.layers.feedforward.recurrent.StepProjection
import shape.komputation.matrix.DoubleMatrix
import shape.komputation.matrix.doubleRowVector

class EncoderStep(
    private val name : String?,
    private val hiddenDimension : Int,
    private val inputProjection: ContinuationLayer,
    private val previousStateProjection: ContinuationLayer,
    private val bias : SeriesBias?,
    private val activationLayer: ContinuationLayer) {

    fun forward(state: DoubleMatrix, input: DoubleMatrix): DoubleMatrix {

        // projected input = input weights * input
        val projectedInput = inputProjection.forward(input)

        // projected state = state weights * state
        val projectedState =  previousStateProjection.forward(state)

        val projectedInputEntries = projectedInput.entries
        val projectedStateEntries = projectedState.entries
        // addition = projected input + projected state
        val additionEntries = add(projectedInputEntries, projectedStateEntries)

        // pre-activation = addition + bias
        val preActivation =

            if(this.bias == null) {

                additionEntries

            }
            else {

                this.bias.forwardStep(additionEntries)

            }


        // activation = activate(pre-activation)
        val newState = activationLayer.forward(DoubleMatrix(hiddenDimension, 1, preActivation))

        return newState

    }

    fun backward(backwardPreviousState : DoubleMatrix?, backwardOutput: DoubleMatrix?): Pair<DoubleMatrix, DoubleMatrix> {

        val backwardAddition : DoubleMatrix

        // Single output encoder at last step
        // Multi-output encoder at last step
        if (backwardPreviousState == null && backwardOutput != null) {

            backwardAddition = backwardOutput
        }
        // Single output encoder at prior steps
        else if (backwardPreviousState != null && backwardOutput == null) {

            backwardAddition = backwardPreviousState

        }
        // Multi-output encoder at prior steps
        else {

            val backwardPreviousStateEntries = backwardPreviousState!!.entries
            val backwardOutputEntries = backwardOutput!!.entries

            backwardAddition = doubleRowVector(*add(backwardPreviousStateEntries, backwardOutputEntries))

        }

        // d activate(state weights * state(1) + input weights * input(2) + bias)) / d state weights * state(1) + input weights * input(2) + bias
        val backwardStateWrtStatePreActivation = this.activationLayer.backward(backwardAddition)

        // d state weights * state(1) + input weights * input(2) + bias / d state(1) = state weights
        val backwardStatePreActivationWrtPreviousState = this.previousStateProjection.backward(backwardStateWrtStatePreActivation)

        // d state weights * state(1) + input weights * input(2) + bias / d input(2) = input weights
        val backwardStatePreActivationWrtInput = this.inputProjection.backward(backwardStateWrtStatePreActivation)

        // d state weights * state(1) + input weights * input(2) + bias / d bias = 1
        this.bias?.backwardStep(backwardStateWrtStatePreActivation)

        return backwardStatePreActivationWrtPreviousState to backwardStatePreActivationWrtInput

    }

}
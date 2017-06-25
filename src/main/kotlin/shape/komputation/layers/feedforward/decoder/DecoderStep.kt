package shape.komputation.layers.feedforward.decoder

import shape.komputation.functions.add
import shape.komputation.layers.ContinuationLayer
import shape.komputation.layers.combination.AdditionCombination
import shape.komputation.layers.feedforward.activation.ActivationLayer
import shape.komputation.layers.feedforward.recurrent.SeriesBias
import shape.komputation.matrix.DoubleMatrix

class DecoderStep(
    private val name : String?,
    private val isLastStep : Boolean,
    private val hiddenDimension : Int,
    private val outputDimension: Int,
    private val inputProjection: ContinuationLayer,
    private val stateProjection : ContinuationLayer,
    private val addition : AdditionCombination,
    private val stateActivation : ActivationLayer,
    private val outputProjection: ContinuationLayer,
    private val outputActivation : ActivationLayer,
    private val bias : SeriesBias?) {

    fun forward(state : DoubleMatrix, input: DoubleMatrix): Pair<DoubleMatrix, DoubleMatrix> {

        // Project the input
        // In the case of a single-input decoder, this is the t-1-th output.
        // In the case of a multi-input decoder, this is the t-th encoding.
        val projectedInput = inputProjection.forward(input)

        // Project the previous state.
        // At te first step in the case of a single-input decoder, this is the encoding.
        // At the first step in the case of a multi-input decoder, this is a zero vector.
        val projectedState =  stateProjection.forward(state)

        // Add the two projections
        val addition = addition.forward(projectedInput, projectedState)

        // Add the bias (if there is one)
        val statePreActivation =

            if(this.bias == null) {

                addition

            }
            else {

                this.bias.forwardStep(addition)

            }

        // Apply the activation function
        val newState = stateActivation.forward(statePreActivation)

        // Project the state
        val outputPreActivation = outputProjection.forward(newState)

        // Apply the activation function to the output pre-activation
        val newOutput = outputActivation.forward(outputPreActivation)

        return newState to newOutput

    }

    fun backward(
        chainStep: DoubleArray,
        backwardStatePreActivationWrtPreviousInput: DoubleMatrix?,
        backwardStatePreActivationWrtPreviousState : DoubleMatrix?): Pair<DoubleMatrix, DoubleMatrix> {

        val outputSum =

            if (isLastStep || backwardStatePreActivationWrtPreviousInput == null) {

                chainStep

            }
            else {

                // Add the gradient with regard to the input from step t+1 (which is the output of step t):
                // d chain / d output(index+1) * d output(index+1) / d input(index + 1) *  d input(index + 1) / d output(index)
                val backwardStatePreactivationWrtPreviousOutputEntries = backwardStatePreActivationWrtPreviousInput.entries

                add(chainStep, backwardStatePreactivationWrtPreviousOutputEntries)

            }

        // Differentiate w.r.t. the output pre-activation:
        // d output / d output pre-activation = d activate(output weights * state) / d output weights * state
        val backwardOutputWrtOutputPreActivation = this.outputActivation.backward(DoubleMatrix(outputDimension, 1, outputSum))

        // Differentiate w.r.t. the state:
        // d output pre-activation (Wh) / d state = d output weights * state / d state
        val backwardOutputPreActivationWrtState = this.outputProjection.backward(backwardOutputWrtOutputPreActivation)

        val stateSum =

            if (isLastStep) {

                backwardOutputPreActivationWrtState.entries

            }
            else {

                val backwardOutputPreActivationWrtStateEntries = backwardOutputPreActivationWrtState.entries

                // d chain / d output(index+1) * d output(index+1) / d state(index)
                val backwardStatePreActivationWrtPreviousStateEntries = backwardStatePreActivationWrtPreviousState!!.entries

                add(backwardOutputPreActivationWrtStateEntries, backwardStatePreActivationWrtPreviousStateEntries)

            }

        // Differentiate w.r.t. the state pre-activation:
        // d state / d state pre-activation = d activate(state weights * state(index-1) + previous output weights * output(index-1) + bias) / d state weights * state(index-1) + previous output weights * output(index-1)
        val backwardStateWrtStatePreActivation = this.stateActivation.backward(DoubleMatrix(hiddenDimension, 1, stateSum))

        // Differentiate w.r.t. the previous output:
        // d state pre-activation / d previous output = d [ state weights * state(index-1) + previous output weights * output(index-1) + bias ] / d output(index-1)
        val newBackwardStatePreActivationWrtInput = this.inputProjection.backward(backwardStateWrtStatePreActivation)

        // Differentiate w.r.t. the previous state:
        // d state pre-activation / d previous state = d [ state weights * state(index-1) + previous output weights * output(index-1) + bias ] / d state(index-1)
        val newBackwardStatePreActivationWrtPreviousState = this.stateProjection.backward(backwardStateWrtStatePreActivation)

        // Differentiate w.r.t. the bias
        this.bias?.backwardStep(backwardStateWrtStatePreActivation)

        return newBackwardStatePreActivationWrtInput to newBackwardStatePreActivationWrtPreviousState

    }

}
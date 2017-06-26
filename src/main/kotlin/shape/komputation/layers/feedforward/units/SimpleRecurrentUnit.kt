package shape.komputation.layers.feedforward.units

import shape.komputation.layers.ContinuationLayer
import shape.komputation.layers.combination.AdditionCombination
import shape.komputation.layers.feedforward.recurrent.SeriesBias
import shape.komputation.matrix.DoubleMatrix

class SimpleRecurrentUnit(
    name : String?,
    private val inputProjection: ContinuationLayer,
    private val previousStateProjection: ContinuationLayer,
    private val addition : AdditionCombination,
    private val bias : SeriesBias?,
    private val activationLayer: ContinuationLayer) : RecurrentUnit(name) {

    override fun forward(state: DoubleMatrix, input: DoubleMatrix): DoubleMatrix {

        // projected input = input weights * input
        val projectedInput = inputProjection.forward(input)

        // projected state = state weights * state
        val projectedState =  previousStateProjection.forward(state)

        // addition = projected input + projected state
        val additionEntries = this.addition.forward(projectedInput, projectedState)

        // pre-activation = addition + bias
        val preActivation =

            if(this.bias == null) {

                additionEntries

            }
            else {

                this.bias.forwardStep(additionEntries)

            }


        // activation = activate(pre-activation)
        val newState = activationLayer.forward(preActivation)

        return newState

    }

    override fun backward(chain: DoubleMatrix): Pair<DoubleMatrix, DoubleMatrix> {

        // d activate(state weights * state(1) + input weights * input(2) + bias)) / d state weights * state(1) + input weights * input(2) + bias
        val backwardStateWrtStatePreActivation = this.activationLayer.backward(chain)

        // d state weights * state(1) + input weights * input(2) + bias / d state(1) = state weights
        val backwardStatePreActivationWrtPreviousState = this.previousStateProjection.backward(backwardStateWrtStatePreActivation)

        // d state weights * state(1) + input weights * input(2) + bias / d input(2) = input weights
        val backwardStatePreActivationWrtInput = this.inputProjection.backward(backwardStateWrtStatePreActivation)

        // d state weights * state(1) + input weights * input(2) + bias / d bias = 1
        this.bias?.backwardStep(backwardStateWrtStatePreActivation)

        return backwardStatePreActivationWrtPreviousState to backwardStatePreActivationWrtInput

    }

}
package shape.komputation.layers.recurrent

import shape.komputation.functions.activation.ActivationFunction
import shape.komputation.functions.backwardProjectionWrtBias
import shape.komputation.initialization.InitializationStrategy
import shape.komputation.initialization.initializeMatrix
import shape.komputation.initialization.initializeRowVector
import shape.komputation.layers.FeedForwardLayer
import shape.komputation.layers.OptimizableLayer
import shape.komputation.layers.feedforward.projection.SharedProjectionLayer
import shape.komputation.layers.feedforward.activation.ActivationLayer
import shape.komputation.layers.feedforward.activation.ReluLayer
import shape.komputation.layers.feedforward.activation.SigmoidLayer
import shape.komputation.layers.feedforward.activation.SoftmaxLayer
import shape.komputation.layers.feedforward.projection.createIdentityProjectionLayer
import shape.komputation.layers.feedforward.projection.createSharedProjectionLayer
import shape.komputation.matrix.*
import shape.komputation.optimization.*
import java.util.*

class RecurrentLayer(
    name : String?,
    private val hiddenDimension : Int,
    private val activationLayers: Array<ActivationLayer>,
    private val stateProjectionLayers : Array<FeedForwardLayer>,
    private val stateWeights : DoubleArray,
    private val stateWeightUpdateRule: UpdateRule? = null,
    private val stateWeightSeriesAccumulator: DenseAccumulator? = null,
    private val stateWeightBatchAccumulator: DenseAccumulator? = null,
    private val inputWeights : DoubleArray,
    private val inputProjectionLayers : Array<SharedProjectionLayer>,
    private val inputWeightUpdateRule: UpdateRule? = null,
    private val inputWeightSeriesAccumulator: DenseAccumulator? = null,
    private val inputWeightBatchAccumulator: DenseAccumulator? = null,
    private val bias: DoubleArray? = null,
    private val biasUpdateRule: UpdateRule? = null) : FeedForwardLayer(name), OptimizableLayer {

    private var state = doubleZeroRowVector(hiddenDimension)

    private val optimizeStateWeights = stateWeightUpdateRule != null
    private val optimizeInputWeights = inputWeightUpdateRule != null
    private val optimizeBias = bias != null && biasUpdateRule != null

    private var input : SequenceMatrix = EMPTY_SEQUENCE_MATRIX

    private var biasSeriesAccumulator = if(optimizeBias) DenseAccumulator(bias!!.size) else null
    private var biasBatchAccumulator = if(optimizeBias) DenseAccumulator(bias!!.size) else null

    override fun forward(input: DoubleMatrix): DoubleMatrix {

        Arrays.fill(state.entries, 0.0)

        input as SequenceMatrix

        this.input = input

        var output : DoubleMatrix? = null

        for (indexStep in 0..input.numberSteps - 1) {

            val step = input.getStep(indexStep)

            // projected input = input weights * input
            val projectedInput = inputProjectionLayers[indexStep].forward(step)
            val projectedInputEntries = projectedInput.entries

            // projected state = state weights * state
            val projectedState = stateProjectionLayers[indexStep].forward(this.state)
            val projectedStateEntries = projectedState.entries

            // addition = projected state + projected input
            val additionEntries = DoubleArray(hiddenDimension) { index ->

                projectedInputEntries[index] + projectedStateEntries[index]

            }

            // pre-activation = addition + bias
            val preActivation =

                if(this.bias == null) {

                    additionEntries

                }
                else {

                    DoubleArray(hiddenDimension) { index ->

                        additionEntries[index] + bias[index]

                    }

                }

            // activation = activate(pre-activation)
            output = activationLayers[indexStep].forward(DoubleMatrix(hiddenDimension, 1, preActivation))

            this.state = output


        }

        return output!!

    }

    override fun backward(chain: DoubleMatrix): DoubleMatrix {

        var seriesChain = chain

        val seriesBackwardWrtInput = zeroSequenceMatrix(this.input.numberSteps, this.input.numberStepRows, this.input.numberStepColumns)

        for (indexStep in this.input.numberSteps - 1 downTo 0) {

            // d activate(state weights * state(1) + input weights * input(2) + bias)) / d state weights * state(1) + input weights * input(2) + bias
            val backwardActivation = this.activationLayers[indexStep].backward(seriesChain)

            // d state weights * state(1) + input weights * input(2) + bias / d state(1) = state weights
            seriesChain = this.stateProjectionLayers[indexStep].backward(backwardActivation)

            // d state weights * state(1) + input weights * input(2) + bias / d input(2) = input weights
            val backwardWrtInput = this.inputProjectionLayers[indexStep].backward(backwardActivation)

            seriesBackwardWrtInput.setStep(indexStep, backwardWrtInput.entries)

            if (this.optimizeBias) {

                // d state weights * state(1) + input weights * input(2) + bias / d bias = 1
                val backwardWrtBias = backwardProjectionWrtBias(this.bias!!.size, seriesChain.entries, seriesChain.numberRows, seriesChain.numberColumns)

                this.biasSeriesAccumulator!!.accumulate(backwardWrtBias)

            }

        }

        if (this.optimizeStateWeights) {

            val stateWeightSeriesAccumulator = this.stateWeightSeriesAccumulator!!

            this.stateWeightBatchAccumulator!!.accumulate(stateWeightSeriesAccumulator.getAccumulation())

            stateWeightSeriesAccumulator.reset()


        }

        if (this.optimizeInputWeights) {

            val inputWeightSeriesAccumulator = this.inputWeightSeriesAccumulator!!

            this.inputWeightBatchAccumulator!!.accumulate(inputWeightSeriesAccumulator.getAccumulation())

            inputWeightSeriesAccumulator.reset()

        }

        if (this.optimizeBias) {

            val seriesAccumulator = this.biasSeriesAccumulator!!
            this.biasBatchAccumulator!!.accumulate(seriesAccumulator.getAccumulation())
            seriesAccumulator.reset()

        }

        return seriesChain

    }

    override fun optimize() {

        if (this.optimizeStateWeights) {

            val batchAccumulator = this.stateWeightBatchAccumulator!!

            updateDensely(this.stateWeights, batchAccumulator.getAccumulation(), batchAccumulator.getCount(), this.stateWeightUpdateRule!!)

            batchAccumulator.reset()

        }

        if (this.optimizeInputWeights) {

            val batchAccumulator = this.inputWeightBatchAccumulator!!

            updateDensely(this.inputWeights, batchAccumulator.getAccumulation(), batchAccumulator.getCount(), this.inputWeightUpdateRule!!)

            batchAccumulator.reset()

        }

        if (this.optimizeBias) {

            val batchAccumulator = this.biasBatchAccumulator!!

            updateDensely(this.bias!!, batchAccumulator.getAccumulation(), batchAccumulator.getCount(), biasUpdateRule!!)

            batchAccumulator.reset()

        }

    }


}


fun createRecurrentLayer(
    numberSteps : Int,
    stepSize : Int,
    hiddenDimension: Int,
    activationFunction : ActivationFunction,
    stateWeightInitializationStrategy: InitializationStrategy,
    inputWeightInitializationStrategy: InitializationStrategy,
    biasInitializationStrategy: InitializationStrategy,
    optimizationStrategy : OptimizationStrategy? = null) =

    createRecurrentLayer(
        null,
        numberSteps,
        stepSize,
        hiddenDimension,
        activationFunction,
        stateWeightInitializationStrategy,
        inputWeightInitializationStrategy,
        biasInitializationStrategy,
        optimizationStrategy)

fun createRecurrentLayer(
    name : String?,
    numberSteps : Int,
    stepSize : Int,
    hiddenDimension: Int,
    activationFunction : ActivationFunction,
    stateWeightInitializationStrategy: InitializationStrategy,
    inputWeightInitializationStrategy: InitializationStrategy,
    biasInitializationStrategy: InitializationStrategy,
    optimizationStrategy : OptimizationStrategy? = null): RecurrentLayer {

    val activationLayers = Array(numberSteps) { index ->

        val activationLayerName = if(name == null) null else "$name-activation-$index"

        when (activationFunction) {

            ActivationFunction.Sigmoid -> SigmoidLayer(activationLayerName)
            ActivationFunction.ReLU -> ReluLayer(activationLayerName)
            ActivationFunction.Softmax -> SoftmaxLayer(activationLayerName)

        }

    }

    val numberStateWeightRows = hiddenDimension
    val numberStateWeightColumns = hiddenDimension
    val stateWeights = initializeMatrix(stateWeightInitializationStrategy, numberStateWeightRows, hiddenDimension)
    val stateWeightSeriesAccumulator = if(optimizationStrategy != null) DenseAccumulator(numberStateWeightRows * numberStateWeightColumns) else null
    val stateWeightBatchAccumulator = if(optimizationStrategy != null) DenseAccumulator(numberStateWeightRows * numberStateWeightColumns) else null
    val stateWeightUpdateRule = if(optimizationStrategy != null) optimizationStrategy(numberStateWeightRows, numberStateWeightColumns) else null

    val stateProjectionLayers = Array(numberSteps) { index ->

        val stateProjectionLayerName = if (name == null) null else "$name-state-projection-$index"

        if (index == 0) {

            createIdentityProjectionLayer(stateProjectionLayerName)

        }
        else {

            createSharedProjectionLayer(stateProjectionLayerName, hiddenDimension, hiddenDimension, stateWeights, stateWeightSeriesAccumulator)
        }

    }

    val numberInputWeightRows = hiddenDimension
    val numberInputWeightColumns = stepSize
    val inputWeights = initializeMatrix(inputWeightInitializationStrategy, numberInputWeightRows, numberInputWeightColumns)
    val inputWeightSeriesAccumulator = if(optimizationStrategy != null) DenseAccumulator(numberInputWeightRows * numberInputWeightColumns) else null
    val inputWeightBatchAccumulator = if(optimizationStrategy != null) DenseAccumulator(numberInputWeightRows * numberInputWeightColumns) else null
    val inputWeightUpdateRule = if(optimizationStrategy != null) optimizationStrategy(numberInputWeightRows, numberInputWeightColumns) else null

    val inputProjectionLayers = Array(numberSteps) { index ->

        val inputProjectionLayerName = if(name == null) null else "$name-input-projection-$index"

        createSharedProjectionLayer(inputProjectionLayerName, stepSize, hiddenDimension, inputWeights, inputWeightSeriesAccumulator)

    }

    val bias = initializeRowVector(biasInitializationStrategy, hiddenDimension)
    val biasUpdateRule = if(optimizationStrategy != null) optimizationStrategy(numberInputWeightRows, numberInputWeightColumns) else null

    return RecurrentLayer(
        name,
        hiddenDimension,
        activationLayers,
        stateProjectionLayers,
        stateWeights,
        stateWeightUpdateRule,
        stateWeightSeriesAccumulator,
        stateWeightBatchAccumulator,
        inputWeights,
        inputProjectionLayers,
        inputWeightUpdateRule,
        inputWeightSeriesAccumulator,
        inputWeightBatchAccumulator,
        bias,
        biasUpdateRule
    )

}
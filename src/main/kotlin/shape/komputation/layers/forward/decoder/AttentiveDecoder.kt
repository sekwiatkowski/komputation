package shape.komputation.layers.forward.decoder

import shape.komputation.functions.activation.ActivationFunction
import shape.komputation.functions.add
import shape.komputation.functions.backwardProjectionWrtInput
import shape.komputation.functions.backwardProjectionWrtWeights
import shape.komputation.functions.extractStep
import shape.komputation.initialization.InitializationStrategy
import shape.komputation.layers.ForwardLayer
import shape.komputation.layers.combination.AdditionCombination
import shape.komputation.layers.combination.additionCombination
import shape.komputation.layers.concatenateNames
import shape.komputation.layers.forward.TranspositionLayer
import shape.komputation.layers.forward.activation.*
import shape.komputation.layers.forward.columnRepetitionLayer
import shape.komputation.layers.forward.projection.*
import shape.komputation.layers.forward.transpositionLayer
import shape.komputation.matrix.*
import shape.komputation.optimization.DenseAccumulator
import shape.komputation.optimization.Optimizable
import shape.komputation.optimization.OptimizationStrategy

class AttentiveDecoder internal constructor(
    private val name : String?,
    private val numberSteps : Int,
    private val encodingDimension: Int,
    private val decodingDimension: Int,
    private val encodingProjection : ProjectionLayer,
    private val attentionPreviousStateWeighting: SeriesWeighting,
    private val columnRepetitions: Array<shape.komputation.layers.forward.ColumnRepetitionLayer>,
    private val attentionAdditions : Array<AdditionCombination>,
    private val tanh: Array<TanhLayer>,
    private val scoringWeighting: SeriesWeighting,
    private val softmax : Array<SoftmaxVectorLayer>,
    private val transposition: Array<TranspositionLayer>,
    private val attendedEncodingWeighting: SeriesWeighting,
    private val decodingPreviousDecoderWeighting: SeriesWeighting,
    private val decodingAdditions: Array<AdditionCombination>,
    private val activationFunctions: Array<ActivationLayer>,
    private val bias : SeriesBias?) : ForwardLayer(name), Optimizable {

    private var attentionDistributionEntries = DoubleArray(this.numberSteps)

    private val encodingSize = this.numberSteps * this.encodingDimension
    private var encodingEntries = DoubleArray(this.encodingSize)
    private val encodingAccumulator = DenseAccumulator(this.encodingSize)

    override fun forward(encodings : DoubleMatrix, isTraining : Boolean): DoubleMatrix {

        var previousDecoderState = doubleZeroColumnVector(this.decodingDimension)
        val output = doubleZeroMatrix(this.encodingDimension, this.numberSteps)

        this.encodingEntries = encodings.entries

        // encoding weights * encodings
        val projectedEncoding = this.encodingProjection.forward(encodings, isTraining)

        val blasEncodingMatrix = createBlasMatrix(this.encodingDimension, this.numberSteps, this.encodingEntries)

        for (indexStep in 0..this.numberSteps - 1) {

            // previous decoder state weights (for attention) * previous decoder state
            val attentionWeightedPreviousState = this.attentionPreviousStateWeighting.forwardStep(indexStep, previousDecoderState, isTraining)

            // expanded weighted previous decoder state (for attention)
            val expandedWeightedPreviousState = this.columnRepetitions[indexStep].forward(attentionWeightedPreviousState, isTraining)

            // pre-activation = projected encodings + expanded weighted previous decoder state (for attention)
            val attentionAddition = this.attentionAdditions[indexStep].forward(projectedEncoding, expandedWeightedPreviousState)

            // attention activation = tanh(pre-activation)
            val attentionActivation = this.tanh[indexStep].forward(attentionAddition, isTraining)

            // unnormalized scores = scoring weights * attention activation << row vector
            val attentionScores = this.scoringWeighting.forwardStep(indexStep, attentionActivation, isTraining)

            // normalized scores = row-wise softmax (unnormalized attention scores) << row vector
            val attentionDistribution = this.softmax[indexStep].forward(attentionScores, isTraining)

            this.attentionDistributionEntries = attentionDistribution.entries

            // normalized scores as a column vector = transposed(normalized scores as a row vector)
            val transposedAttentionDistribution = this.transposition[indexStep].forward(attentionDistribution, isTraining)

            // attended encoding = encodings * normalized scores as column vector
            val blasTransposedAttentionDistribution = createBlasMatrix(this.numberSteps, 1, transposedAttentionDistribution.entries)

            val attendedEncoding = DoubleMatrix(encodingDimension, 1, blasEncodingMatrix.multiply(blasTransposedAttentionDistribution).getEntries())

            // weighted attended encoding = attended encoding weights * attended encoding
            val weightedAttendedEncoding = this.attendedEncodingWeighting.forwardStep(indexStep, attendedEncoding, isTraining)

            // previous decoder state weights (for decoding) * previous decoder state weights
            val decodingWeightedPreviousState = this.decodingPreviousDecoderWeighting.forwardStep(indexStep, previousDecoderState, isTraining)

            // weighted attended encoding + decoding weighted previous state
            val decodingAddition = this.decodingAdditions[indexStep].forward(weightedAttendedEncoding, decodingWeightedPreviousState)

            val newDecoderStatePreActivation =

                if(this.bias == null)
                    decodingAddition
                else
                    this.bias.forwardStep(decodingAddition)

            val newDecoderState = this.activationFunctions[indexStep].forward(newDecoderStatePreActivation, isTraining)

            output.setColumn(indexStep, newDecoderState.entries)

            previousDecoderState = newDecoderState

        }

        return output

    }

    override fun backward(chain: DoubleMatrix): DoubleMatrix {

        val chainEntries = chain.entries

        var diffNextDecoderStateWrtDecoderState : DoubleArray? = null

        for (indexStep in this.numberSteps - 1 downTo 0) {

            val isLastStep = indexStep + 1 == this.numberSteps

            val diffOutputWrtDecoderState = extractStep(chainEntries, indexStep, this.decodingDimension)

            val sumWrtDecoderState = doubleColumnVector(*(
                if (isLastStep) {

                    diffOutputWrtDecoderState

                }
                else {

                    add(diffOutputWrtDecoderState, diffNextDecoderStateWrtDecoderState!!)

                }))

            // f'(U_a * Ea^T + U_d * d_t-1 + bias) = d f(U_a * Ea^T + U_d * d_t-1 + bias) / d (U_a * Ea^T + U_d * d_t-1 + bias)
            val diffDecodingWrtDecodingPreActivation = this.activationFunctions[indexStep].backward(sumWrtDecoderState)

            // d (U_a * Ea^T + U_d * d_t-1 + bias) / d Ea^T
            val diffPreActivationWrtAttendedEncoding = this.attendedEncodingWeighting.backwardStep(indexStep, diffDecodingWrtDecodingPreActivation)

            // d (U_a * Ea^T + U_d * d_t-1 + bias) / d d_t-1
            val diffPreActivationWrtWeightedPreviousStateForDecoding = this.decodingPreviousDecoderWeighting.backwardStep(indexStep, diffDecodingWrtDecodingPreActivation)

            this.bias?.backwardStep(diffDecodingWrtDecodingPreActivation)

            /* Ea^T
                                        a_1
                                        a_2
                                        a_3
                e(1)_1 e(2)_1 e(3)_1    e(1)_1 * a_1 + e(2)_1 * a_2 + e(3)_1 * a_3
                e(1)_2 e(2)_2 e(3)_2    e(1)_2 * a_1 + e(2)_2 * a_2 + e(3)_2 * a_3*/

            /* d Ea^t / d a^t
               d Ea^t / d a_(1) = e(1)_1 + e(1)_2
               d Ea^t / d a_(2) = e(2)_1 + e(2)_2
               d Ea^t / d a_(3) = e(3)_1 + e(3)_2*/

            val diffPreActivationWrtAttendedEncodingEntries = diffPreActivationWrtAttendedEncoding.entries
            val diffPreActivationWrtAttendedEncodingNumberRows = diffPreActivationWrtAttendedEncoding.numberRows
            val diffPreActivationWrtAttendedEncodingNumberColumns = diffPreActivationWrtAttendedEncoding.numberColumns

            val diffAttendedEncodingWrtTransposedAttentionDistributionEntries = backwardProjectionWrtInput(
                this.numberSteps,
                1,
                this.numberSteps,
                this.encodingEntries,
                this.encodingDimension,
                diffPreActivationWrtAttendedEncodingEntries,
                diffPreActivationWrtAttendedEncodingNumberRows)
            val diffAttendedEncodingWrtTransposedAttentionDistribution = DoubleMatrix(numberSteps, 1, diffAttendedEncodingWrtTransposedAttentionDistributionEntries)

            /* d Ea^t / d E
               d Ea^t / d e(1)_1 = a_1
               d Ea^t / d e(1)_2 = a_1 */

            val diffAttendedEncodingWrtEncoding = backwardProjectionWrtWeights(
                this.encodingSize,
                this.encodingDimension,
                this.numberSteps,
                this.attentionDistributionEntries,
                this.numberSteps,
                diffPreActivationWrtAttendedEncodingEntries,
                diffPreActivationWrtAttendedEncodingNumberRows,
                diffPreActivationWrtAttendedEncodingNumberColumns)

            this.encodingAccumulator.accumulate(diffAttendedEncodingWrtEncoding)

            // d a^T / d a = d a^T / d softmax(pre-activation)
            val diffWrtTransposedAttentionDistributionWrtAttentionDistribution = this.transposition[indexStep].backward(diffAttendedEncodingWrtTransposedAttentionDistribution)

            // d softmax(pre-activation) / d pre-activation
            val diffAttentionDistributionWrtAttentionScores = this.softmax[indexStep].backward(diffWrtTransposedAttentionDistributionWrtAttentionDistribution)

            // d s * tanh(...) / d tanh (...)
            val diffAttentionScoresWrtAttentionActivation = this.scoringWeighting.backwardStep(indexStep, diffAttentionDistributionWrtAttentionScores)

            // d tanh(...) / d W^e * E + expand(...)
            val diffAttentionActivationWrtAttentionPreactivation = this.tanh[indexStep].backward(diffAttentionScoresWrtAttentionActivation)

            // d W^e * E + expand(...) / d E
            val diffAttentionPreactivationWrtEncodings = this.encodingProjection.backward(diffAttentionActivationWrtAttentionPreactivation)
            this.encodingAccumulator.accumulate(diffAttentionPreactivationWrtEncodings.entries)

            // d W^e * E + expand(W^d * d_t-1) / d W^d * d_t-1
            val diffAttentionPreactivationWrtExpansion = this.columnRepetitions[indexStep].backward(diffAttentionActivationWrtAttentionPreactivation)

            //  d W^d * d_t-1 / d d_t-1
            val diffExpansionWrtWeightedPreviousState = this.attentionPreviousStateWeighting.backwardStep(indexStep, diffAttentionPreactivationWrtExpansion)

            diffNextDecoderStateWrtDecoderState = diffPreActivationWrtWeightedPreviousStateForDecoding.entries + diffExpansionWrtWeightedPreviousState.entries

        }

        // W^e is used once per series.

        // W^d
        this.attentionPreviousStateWeighting.backwardSeries()
        // s
        this.scoringWeighting.backwardSeries()
        // U^e
        this.attendedEncodingWeighting.backwardSeries()
        // U^d
        this.decodingPreviousDecoderWeighting.backwardSeries()
        // b
        this.bias?.backwardSeries()

        val encodingAccumulation = encodingAccumulator.getAccumulation().copyOf()
        val result = DoubleMatrix(this.encodingDimension, this.numberSteps, encodingAccumulation)

        this.encodingAccumulator.reset()

        return result

    }

    override fun optimize(scalingFactor : Double) {

        // W^e
        this.encodingProjection.optimize(scalingFactor)
        // W^d
        this.attentionPreviousStateWeighting.optimize(scalingFactor)
        // s
        this.scoringWeighting.optimize(scalingFactor)
        // U^e
        this.attendedEncodingWeighting.optimize(scalingFactor)
        // U^d
        this.decodingPreviousDecoderWeighting.optimize(scalingFactor)
        // b
        this.bias?.optimize(scalingFactor)

    }

}

fun attentiveDecoder(
    numberSteps : Int,
    encodingDimension : Int,
    decodingDimension: Int,
    activationFunction: ActivationFunction,
    weightInitializationStrategy: InitializationStrategy,
    biasInitializationStrategy: InitializationStrategy?,
    optimizationStrategy: OptimizationStrategy) =

    attentiveDecoder(
        null,
        numberSteps,
        encodingDimension,
        decodingDimension,
        activationFunction,
        weightInitializationStrategy,
        biasInitializationStrategy,
        optimizationStrategy
    )

fun attentiveDecoder(
    name : String?,
    numberSteps : Int,
    encodingDimension : Int,
    decodingDimension: Int,
    activationFunction: ActivationFunction,
    weightInitializationStrategy: InitializationStrategy,
    biasInitializationStrategy: InitializationStrategy?,
    optimizationStrategy: OptimizationStrategy): AttentiveDecoder {

    val encodingProjectionName = concatenateNames(name, "encoding-projection")
    val encodingProjection = projectionLayer(encodingProjectionName, encodingDimension, encodingDimension, weightInitializationStrategy, null, optimizationStrategy)

    val columnRepetitionLayers = Array(numberSteps) { indexStep ->

        val columnRepetitionLayerName = concatenateNames(name, "column-repetition-$indexStep")

        columnRepetitionLayer(columnRepetitionLayerName, numberSteps)
    }

    val attentionAdditions = Array(numberSteps) { indexStep ->

        val attentionAdditionName = concatenateNames(name, "attention-addition-$indexStep")

        additionCombination(attentionAdditionName)

    }

    val attentionPreviousStateWeightingSeriesName = concatenateNames(name, "attention-previous-state-weighting")
    val attentionPreviousStateWeightingStepName = concatenateNames(name, "attention-previous-state-weighting-step")
    val attentionPreviousStateWeighting = seriesWeighting(attentionPreviousStateWeightingSeriesName, attentionPreviousStateWeightingStepName, numberSteps, true, decodingDimension, encodingDimension, weightInitializationStrategy, optimizationStrategy)

    val tanh = Array(numberSteps) { tanhLayer() }

    val scoringWeightingSeriesName = concatenateNames(name, "scoring-weighting")
    val scoringWeightingStepName = concatenateNames(name, "scoring-weighting-step")
    val scoringWeighting = seriesWeighting(scoringWeightingSeriesName, scoringWeightingStepName, numberSteps, false, encodingDimension, 1, weightInitializationStrategy, optimizationStrategy)

    val softmax = Array(numberSteps) { softmaxVectorLayer() }

    val transposition = Array(numberSteps) { transpositionLayer() }

    val attendedEncodingWeightingSeriesName = concatenateNames(name, "attended-encoding-weighting")
    val attendedEncodingWeightingStepName = concatenateNames(name, "attended-encoding-weighting-step")
    val attendedEncodingWeighting = seriesWeighting(attendedEncodingWeightingSeriesName, attendedEncodingWeightingStepName, numberSteps, false, encodingDimension, encodingDimension, weightInitializationStrategy, optimizationStrategy)

    val decodingPreviousStateWeightingSeriesName = concatenateNames(name, "decoding-previous-state-weighting")
    val decodingPreviousStateWeightingStepName = concatenateNames(name, "decoding-previous-state-weighting-step")
    val decodingPreviousStateWeighting = seriesWeighting(decodingPreviousStateWeightingSeriesName, decodingPreviousStateWeightingStepName, numberSteps, true, decodingDimension, decodingDimension, weightInitializationStrategy, optimizationStrategy)

    val decodingAdditions = Array(numberSteps) { indexStep ->

        val decodingAdditionName = concatenateNames(name, "decoding-addition-$indexStep")

        AdditionCombination(decodingAdditionName)

    }

    val activationName = concatenateNames(name, "decoding-activation")
    val activation = activationLayers(numberSteps, activationName, activationFunction)

    val bias =
        if(biasInitializationStrategy == null)
            null
        else {

            val biasName =  concatenateNames(name, "bias")

            seriesBias(biasName, decodingDimension, biasInitializationStrategy, optimizationStrategy)
        }

    val attentiveDecoder = AttentiveDecoder(
        name,
        numberSteps,
        encodingDimension,
        decodingDimension,
        encodingProjection,
        attentionPreviousStateWeighting,
        columnRepetitionLayers,
        attentionAdditions,
        tanh,
        scoringWeighting,
        softmax,
        transposition,
        attendedEncodingWeighting,
        decodingPreviousStateWeighting,
        decodingAdditions,
        activation,
        bias
    )

    return attentiveDecoder

}

package shape.komputation.layers.feedforward.decoder

import shape.komputation.functions.*
import shape.komputation.functions.activation.ActivationFunction
import shape.komputation.initialization.InitializationStrategy
import shape.komputation.layers.ContinuationLayer
import shape.komputation.layers.OptimizableLayer
import shape.komputation.layers.concatenateNames
import shape.komputation.layers.feedforward.ColumnRepetitionLayer
import shape.komputation.layers.feedforward.TranspositionLayer
import shape.komputation.layers.feedforward.activation.ActivationLayer
import shape.komputation.layers.feedforward.activation.SoftmaxVectorLayer
import shape.komputation.layers.feedforward.activation.TanhLayer
import shape.komputation.layers.feedforward.activation.createActivationLayers
import shape.komputation.layers.feedforward.projection.ProjectionLayer
import shape.komputation.layers.feedforward.projection.createProjectionLayer
import shape.komputation.layers.feedforward.recurrent.SeriesBias
import shape.komputation.layers.feedforward.recurrent.SeriesProjection
import shape.komputation.layers.feedforward.recurrent.createSeriesBias
import shape.komputation.layers.feedforward.recurrent.createSeriesProjection
import shape.komputation.matrix.*
import shape.komputation.optimization.DenseAccumulator
import shape.komputation.optimization.OptimizationStrategy

class AttentiveDecoder(
    name : String?,
    private val numberSteps : Int,
    private val encodingDimension: Int,
    private val decodingDimension: Int,
    private val columnRepetitionLayers: Array<ColumnRepetitionLayer>,
    private val encodingProjection : ProjectionLayer,
    private val attentionPreviousStateProjection: SeriesProjection,
    private val tanh: Array<TanhLayer>,
    private val scoringProjection : SeriesProjection,
    private val softmax : Array<SoftmaxVectorLayer>,
    private val transposition: Array<TranspositionLayer>,
    private val attendedEncodingProjection: SeriesProjection,
    private val decodingPreviousDecoderProjection: SeriesProjection,
    private val activationFunctions: Array<ActivationLayer>,
    private val bias : SeriesBias?) : ContinuationLayer(name), OptimizableLayer {

    private var attentionDistributionEntries = DoubleArray(this.numberSteps)

    private val encodingSize = this.numberSteps * this.encodingDimension
    private var encodingEntries = DoubleArray(this.encodingSize)
    private val encodingAccumulator = DenseAccumulator(this.encodingSize)

    override fun forward(encodings : DoubleMatrix): DoubleMatrix {

        encodings as SequenceMatrix

        var previousDecoderState = doubleZeroColumnVector(this.decodingDimension)
        val output = zeroSequenceMatrix(this.numberSteps, this.encodingDimension)

        this.encodingEntries = encodings.entries

        // encoding weights * encodings
        val projectedEncoding = this.encodingProjection.forward(encodings)

        val blasEncodingMatrix = createBlasMatrix(this.encodingDimension, this.numberSteps, this.encodingEntries)

        for (indexStep in 0..this.numberSteps - 1) {

            // previous decoder state weights (for attention) * previous decoder state
            val attentionProjectedPreviousState = this.attentionPreviousStateProjection.forwardStep(indexStep, previousDecoderState)

            // expanded projected previous decoder state (for attention)
            val expandedProjectedPreviousState = this.columnRepetitionLayers[indexStep].forward(attentionProjectedPreviousState)

            // pre-activation = projected encodings + expanded projected previous decoder state (for attention)
            val attentionAdditionEntries = add(projectedEncoding.entries, expandedProjectedPreviousState.entries)

            // attention activation = tanh(pre-activation)
            val attentionAddition = DoubleMatrix(this.encodingDimension, this.numberSteps, attentionAdditionEntries)
            val attentionActivation = this.tanh[indexStep].forward(attentionAddition)

            // unnormalized scores = scoring weights * attention activation << row vector
            val attentionScores = this.scoringProjection.forwardStep(indexStep, attentionActivation)

            // normalized scores = row-wise softmax (unnormalized attention scores) << row vector
            val attentionDistribution = this.softmax[indexStep].forward(attentionScores)

            this.attentionDistributionEntries = attentionDistribution.entries

            // normalized scores as a column vector = transposed(normalized scores as a row vector)
            val transposedAttentionDistribution = this.transposition[indexStep].forward(attentionDistribution)

            // attended encoding = encodings * normalized scores as column vector
            val blasTransposedAttentionDistribution = createBlasMatrix(this.numberSteps, 1, transposedAttentionDistribution.entries)

            val attendedEncoding = DoubleMatrix(encodingDimension, 1, blasEncodingMatrix.multiply(blasTransposedAttentionDistribution).getEntries())

            // projected attended encoding = attended encoding weights * attended encoding
            val projectedAttendedEncoding = this.attendedEncodingProjection.forwardStep(indexStep, attendedEncoding)

            // previous decoder state weights (for decoding) * previous decoder state weights
            val decodingProjectedPreviousState = this.decodingPreviousDecoderProjection.forwardStep(indexStep, previousDecoderState)

            // projectedAttendedEncoding + decodingProjectedPreviousState
            val decodingAdditionEntries = add(projectedAttendedEncoding.entries, decodingProjectedPreviousState.entries)

            val newDecoderStatePreActivation = doubleColumnVector(*(
                if(this.bias == null)
                    decodingAdditionEntries
                else
                    this.bias.forwardStep(decodingAdditionEntries)
                ))

            val newDecoderState = this.activationFunctions[indexStep].forward(newDecoderStatePreActivation)

            output.setStep(indexStep, newDecoderState.entries)

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
            val diffPreActivationWrtAttendedEncoding = this.attendedEncodingProjection.backwardStep(indexStep, diffDecodingWrtDecodingPreActivation)

            // d (U_a * Ea^T + U_d * d_t-1 + bias) / d d_t-1
            val diffPreActivationWrtProjectedPreviousStateForDecoding = this.decodingPreviousDecoderProjection.backwardStep(indexStep, diffDecodingWrtDecodingPreActivation)

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
            val diffAttentionScoresWrtAttentionActivation = this.scoringProjection.backwardStep(indexStep, diffAttentionDistributionWrtAttentionScores)

            // d tanh(...) / d W^e * E + expand(...)
            val diffAttentionActivationWrtAttentionPreactivation = this.tanh[indexStep].backward(diffAttentionScoresWrtAttentionActivation)

            // d W^e * E + expand(...) / d E
            val diffAttentionPreactivationWrtEncodings = this.encodingProjection.backward(diffAttentionActivationWrtAttentionPreactivation)
            this.encodingAccumulator.accumulate(diffAttentionPreactivationWrtEncodings.entries)

            // d W^e * E + expand(W^d * d_t-1) / d W^d * d_t-1
            val diffAttentionPreactivationWrtExpansion = this.columnRepetitionLayers[indexStep].backward(diffAttentionActivationWrtAttentionPreactivation)

            //  d W^d * d_t-1 / d d_t-1
            val diffExpansionWrtProjectedPreviousStateForAttention = this.attentionPreviousStateProjection.backwardStep(indexStep, diffAttentionPreactivationWrtExpansion)

            diffNextDecoderStateWrtDecoderState = diffPreActivationWrtProjectedPreviousStateForDecoding.entries + diffExpansionWrtProjectedPreviousStateForAttention.entries

        }

        // W^e is used once per series.

        // W^d
        this.attentionPreviousStateProjection.backwardSeries()
        // s
        this.scoringProjection.backwardSeries()
        // U^e
        this.attendedEncodingProjection.backwardSeries()
        // U^d
        this.decodingPreviousDecoderProjection.backwardSeries()
        // b
        this.bias?.backwardSeries()

        val encodingAccumulation = encodingAccumulator.getAccumulation().copyOf()
        val result = DoubleMatrix(this.encodingDimension, this.numberSteps, encodingAccumulation)

        encodingAccumulator.reset()

        return result

    }

    override fun optimize() {

        // W^e
        this.encodingProjection.optimize()
        // W^d
        this.attentionPreviousStateProjection.optimize()
        // s
        this.scoringProjection.optimize()
        // U^e
        this.attendedEncodingProjection.optimize()
        // U^d
        this.decodingPreviousDecoderProjection.optimize()
        // b
        this.bias?.optimize()

    }

}

fun createAttentiveDecoder(
    name : String?,
    numberSteps : Int,
    encodingDimension : Int,
    decodingDimension: Int,
    activationFunction: ActivationFunction,
    weightInitializationStrategy: InitializationStrategy,
    biasInitializationStrategy: InitializationStrategy?,
    optimizationStrategy: OptimizationStrategy): AttentiveDecoder {

    val columnRepetitionLayers = Array(numberSteps) { indexStep ->

        val columnRepetitionLayerName = concatenateNames(name, "column-repetition-$indexStep")

        ColumnRepetitionLayer(columnRepetitionLayerName, numberSteps)
    }

    val encodingProjectionName = concatenateNames(name, "encoding-projection")
    val encodingProjection = createProjectionLayer(encodingProjectionName, encodingDimension, encodingDimension, false, weightInitializationStrategy, optimizationStrategy)

    val attentionPreviousStateProjectionName = concatenateNames(name, "attention-previous-state-projection")
    val (attentionPreviousStateSeriesProjection, _) = createSeriesProjection(attentionPreviousStateProjectionName, numberSteps, true, decodingDimension, encodingDimension, weightInitializationStrategy, optimizationStrategy)

    val tanh = Array(numberSteps) { TanhLayer() }

    val scoringProjectionName = concatenateNames(name, "scoring-projection")
    val (scoringProjection, _) = createSeriesProjection(scoringProjectionName, numberSteps, false, encodingDimension, 1, weightInitializationStrategy, optimizationStrategy)

    val softmax = Array(numberSteps) { SoftmaxVectorLayer() }

    val transposition = Array(numberSteps) { TranspositionLayer() }

    val attendedEncodingProjectionName = concatenateNames(name, "attended-encoding-projection")
    val (attendedEncodingProjection, _) = createSeriesProjection(attendedEncodingProjectionName, numberSteps, false, encodingDimension, encodingDimension, weightInitializationStrategy, optimizationStrategy)

    val decodingPreviousStateProjectionName = concatenateNames(name, "decoding-previous-state-projection")
    val (decodedPreviousStateProjection, _) = createSeriesProjection(decodingPreviousStateProjectionName, numberSteps, true, decodingDimension, decodingDimension, weightInitializationStrategy, optimizationStrategy)

    val activationName = concatenateNames(name, "decoding-activation")
    val activation = createActivationLayers(numberSteps, activationName, activationFunction)

    val bias =
        if(biasInitializationStrategy == null)
            null
        else {

            val biasName =  concatenateNames(name, "bias")

            createSeriesBias(biasName, decodingDimension, biasInitializationStrategy, optimizationStrategy)
        }

    val attentiveDecoder = AttentiveDecoder(
        name,
        numberSteps,
        encodingDimension,
        decodingDimension,
        columnRepetitionLayers,
        encodingProjection,
        attentionPreviousStateSeriesProjection,
        tanh,
        scoringProjection,
        softmax,
        transposition,
        attendedEncodingProjection,
        decodedPreviousStateProjection,
        activation,
        bias
    )

    return attentiveDecoder

}

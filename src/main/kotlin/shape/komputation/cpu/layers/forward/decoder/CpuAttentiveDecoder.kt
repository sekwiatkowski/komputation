package shape.komputation.cpu.layers.forward.decoder

import shape.komputation.cpu.functions.*
import shape.komputation.cpu.layers.BaseCpuForwardLayer
import shape.komputation.cpu.layers.combination.AdditionCombination
import shape.komputation.cpu.layers.forward.CpuColumnRepetitionLayer
import shape.komputation.cpu.layers.forward.CpuTranspositionLayer
import shape.komputation.cpu.layers.forward.activation.CpuActivationLayer
import shape.komputation.cpu.layers.forward.activation.CpuSoftmaxLayer
import shape.komputation.cpu.layers.forward.activation.CpuTanhLayer
import shape.komputation.cpu.layers.forward.projection.CpuProjectionLayer
import shape.komputation.cpu.layers.forward.projection.SeriesBias
import shape.komputation.cpu.layers.forward.projection.SeriesWeighting
import shape.komputation.cpu.optimization.DenseAccumulator
import shape.komputation.matrix.FloatMatrix
import shape.komputation.matrix.floatColumnVector
import shape.komputation.matrix.floatZeroColumnVector
import shape.komputation.optimization.Optimizable

class CpuAttentiveDecoder internal constructor(
    name : String?,
    private val numberSteps : Int,
    private val encodingDimension: Int,
    private val decodingDimension: Int,
    private val encodingProjection : CpuProjectionLayer,
    private val attentionPreviousStateWeighting: SeriesWeighting,
    private val columnRepetitions: Array<CpuColumnRepetitionLayer>,
    private val attentionAdditions : Array<AdditionCombination>,
    private val tanh: Array<CpuTanhLayer>,
    private val scoringWeighting: SeriesWeighting,
    private val softmax : Array<CpuSoftmaxLayer>,
    private val transposition: Array<CpuTranspositionLayer>,
    private val attendedEncodingWeighting: SeriesWeighting,
    private val decodingPreviousDecoderWeighting: SeriesWeighting,
    private val decodingAdditions: Array<AdditionCombination>,
    private val bias : SeriesBias?,
    private val activations: Array<CpuActivationLayer>) : BaseCpuForwardLayer(name), Optimizable {

    private var attentionDistributionEntries = FloatArray(this.numberSteps)

    private val encodingSize = this.numberSteps * this.encodingDimension
    private var encodingEntries = FloatArray(this.encodingSize)
    private val encodingAccumulator = DenseAccumulator(this.encodingSize)

    private val outputEntries = FloatArray(this.decodingDimension * this.numberSteps)

    override fun forward(withinBatch : Int, encodings : FloatMatrix, isTraining : Boolean): FloatMatrix {

        var previousDecoderState = floatZeroColumnVector(this.decodingDimension)

        this.encodingEntries = encodings.entries

        // encoding weights * encodings
        val projectedEncoding = this.encodingProjection.forward(withinBatch, encodings, isTraining)

        val blasEncodingMatrix = org.jblas.FloatMatrix(this.encodingDimension, this.numberSteps, *this.encodingEntries)

        for (indexStep in 0..this.numberSteps - 1) {

            // previous decoder state weights (for attention) * previous decoder state
            val attentionWeightedPreviousState = this.attentionPreviousStateWeighting.forwardStep(withinBatch, indexStep, previousDecoderState, isTraining)

            // expanded weighted previous decoder state (for attention)
            val expandedWeightedPreviousState = this.columnRepetitions[indexStep].forward(withinBatch, attentionWeightedPreviousState, isTraining)

            // pre-activation = projected encodings + expanded weighted previous decoder state (for attention)
            val attentionAddition = this.attentionAdditions[indexStep].forward(projectedEncoding, expandedWeightedPreviousState)

            // attention activation = tanh(pre-activation)
            val attentionActivation = this.tanh[indexStep].forward(withinBatch, attentionAddition, isTraining)

            // unnormalized scores = scoring weights * attention activation << row vector
            val attentionScores = this.scoringWeighting.forwardStep(withinBatch, indexStep, attentionActivation, isTraining)

            // normalized scores = row-wise softmax (unnormalized attention scores) << row vector
            val attentionDistribution = this.softmax[indexStep].forward(withinBatch, attentionScores, isTraining)

            this.attentionDistributionEntries = attentionDistribution.entries

            // normalized scores as a column vector = transposed(normalized scores as a row vector)
            val transposedAttentionDistribution = this.transposition[indexStep].forward(withinBatch, attentionDistribution, isTraining)

            // attended encoding = encodings * normalized scores as column vector
            val blasTransposedAttentionDistribution = org.jblas.FloatMatrix(this.numberSteps, 1, *transposedAttentionDistribution.entries)

            val blasAttendedEncoding = org.jblas.FloatMatrix(this.encodingDimension, 1)

            multiply(
                blasEncodingMatrix,
                blasTransposedAttentionDistribution,
                blasAttendedEncoding
            )

            val attendedEncoding = FloatMatrix(this.encodingDimension, 1, blasAttendedEncoding.data)

            // weighted attended encoding = attended encoding weights * attended encoding
            val weightedAttendedEncoding = this.attendedEncodingWeighting.forwardStep(withinBatch, indexStep, attendedEncoding, isTraining)

            // previous decoder state weights (for decoding) * previous decoder state weights
            val decodingWeightedPreviousState = this.decodingPreviousDecoderWeighting.forwardStep(withinBatch, indexStep, previousDecoderState, isTraining)

            // weighted attended encoding + decoding weighted previous state
            val decodingAddition = this.decodingAdditions[indexStep].forward(weightedAttendedEncoding, decodingWeightedPreviousState)

            val newDecoderStatePreActivation =

                if(this.bias == null)
                    decodingAddition
                else
                    this.bias.forwardStep(withinBatch, indexStep, decodingAddition, isTraining)

            val newDecoderState = this.activations[indexStep].forward(withinBatch, newDecoderStatePreActivation, isTraining)

            setStep(newDecoderState.entries, indexStep, this.outputEntries, this.decodingDimension)

            previousDecoderState = newDecoderState

        }

        return FloatMatrix(this.decodingDimension, this.numberSteps, this.outputEntries)

    }

    private val sumWrtDecoderStateEntries = FloatArray(this.decodingDimension)
    private val diffOutputWrtDecoderState = FloatArray(this.decodingDimension)

    private val diffAttendedEncodingWrtEncoding = FloatArray(this.encodingSize)

    override fun backward(withinBatch : Int, chain: FloatMatrix): FloatMatrix {

        val chainEntries = chain.entries

        var diffNextDecoderStateWrtDecoderState : FloatArray? = null

        for (indexStep in this.numberSteps - 1 downTo 0) {

            val isLastStep = indexStep + 1 == this.numberSteps

            getStep(chainEntries, indexStep, diffOutputWrtDecoderState, this.decodingDimension)

            val sumWrtDecoderState = floatColumnVector(*(
                if (isLastStep) {

                    diffOutputWrtDecoderState

                } else {

                    add(diffOutputWrtDecoderState, diffNextDecoderStateWrtDecoderState!!, this.sumWrtDecoderStateEntries, this.decodingDimension)

                    this.sumWrtDecoderStateEntries

                }))

            // f'(U_a * Ea^T + U_d * d_t-1 + bias) = d f(U_a * Ea^T + U_d * d_t-1 + bias) / d (U_a * Ea^T + U_d * d_t-1 + bias)
            val diffDecodingWrtDecodingPreActivation = this.activations[indexStep].backward(withinBatch, sumWrtDecoderState)

            // d (U_a * Ea^T + U_d * d_t-1 + bias) / d Ea^T
            val diffPreActivationWrtAttendedEncoding = this.attendedEncodingWeighting.backwardStep(withinBatch, indexStep, diffDecodingWrtDecodingPreActivation)

            // d (U_a * Ea^T + U_d * d_t-1 + bias) / d d_t-1
            val diffPreActivationWrtWeightedPreviousStateForDecoding = this.decodingPreviousDecoderWeighting.backwardStep(withinBatch, indexStep, diffDecodingWrtDecodingPreActivation)

            this.bias?.backwardStep(withinBatch, indexStep, diffDecodingWrtDecodingPreActivation)

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

            val diffAttendedEncodingWrtTransposedAttentionDistributionEntries = FloatArray(this.numberSteps)

            backwardProjectionWrtInput(
                this.numberSteps,
                1,
                this.encodingEntries,
                this.encodingDimension,
                diffPreActivationWrtAttendedEncodingEntries,
                diffPreActivationWrtAttendedEncodingNumberRows,
                diffAttendedEncodingWrtTransposedAttentionDistributionEntries)

            val diffAttendedEncodingWrtTransposedAttentionDistribution = FloatMatrix(numberSteps, 1, diffAttendedEncodingWrtTransposedAttentionDistributionEntries)

            /* d Ea^t / d E
               d Ea^t / d e(1)_1 = a_1
               d Ea^t / d e(1)_2 = a_1 */

            backwardProjectionWrtWeights(
                this.encodingDimension,
                this.numberSteps,
                this.attentionDistributionEntries,
                this.numberSteps,
                diffPreActivationWrtAttendedEncodingEntries,
                diffPreActivationWrtAttendedEncodingNumberRows,
                diffPreActivationWrtAttendedEncodingNumberColumns,
                this.diffAttendedEncodingWrtEncoding)

            this.encodingAccumulator.accumulate(this.diffAttendedEncodingWrtEncoding)

            // d a^T / d a = d a^T / d softmax(pre-activation)
            val diffWrtTransposedAttentionDistributionWrtAttentionDistribution = this.transposition[indexStep].backward(withinBatch, diffAttendedEncodingWrtTransposedAttentionDistribution)

            // d softmax(pre-activation) / d pre-activation
            val diffAttentionDistributionWrtAttentionScores = this.softmax[indexStep].backward(withinBatch, diffWrtTransposedAttentionDistributionWrtAttentionDistribution)

            // d s * tanh(...) / d tanh (...)
            val diffAttentionScoresWrtAttentionActivation = this.scoringWeighting.backwardStep(withinBatch, indexStep, diffAttentionDistributionWrtAttentionScores)

            // d tanh(...) / d W^e * E + expand(...)
            val diffAttentionActivationWrtAttentionPreactivation = this.tanh[indexStep].backward(withinBatch, diffAttentionScoresWrtAttentionActivation)

            // d W^e * E + expand(...) / d E
            val diffAttentionPreactivationWrtEncodings = this.encodingProjection.backward(withinBatch, diffAttentionActivationWrtAttentionPreactivation)
            this.encodingAccumulator.accumulate(diffAttentionPreactivationWrtEncodings.entries)

            // d W^e * E + expand(W^d * d_t-1) / d W^d * d_t-1
            val diffAttentionPreactivationWrtExpansion = this.columnRepetitions[indexStep].backward(withinBatch, diffAttentionActivationWrtAttentionPreactivation)

            //  d W^d * d_t-1 / d d_t-1
            val diffExpansionWrtWeightedPreviousState = this.attentionPreviousStateWeighting.backwardStep(withinBatch, indexStep, diffAttentionPreactivationWrtExpansion)

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
        val result = FloatMatrix(this.encodingDimension, this.numberSteps, encodingAccumulation)

        this.encodingAccumulator.reset()

        return result

    }

    override fun optimize(scalingFactor : Float) {

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
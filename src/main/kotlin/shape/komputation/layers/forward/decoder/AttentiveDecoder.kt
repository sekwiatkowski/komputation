package shape.komputation.layers.forward.decoder

import shape.komputation.cpu.layers.combination.AdditionCombination
import shape.komputation.cpu.layers.combination.additionCombination
import shape.komputation.cpu.layers.forward.activation.cpuActivationLayer
import shape.komputation.cpu.layers.forward.decoder.CpuAttentiveDecoder
import shape.komputation.cpu.layers.forward.projection.seriesBias
import shape.komputation.cpu.layers.forward.projection.seriesWeighting
import shape.komputation.initialization.InitializationStrategy
import shape.komputation.layers.CpuForwardLayerInstruction
import shape.komputation.layers.concatenateNames
import shape.komputation.layers.forward.activation.ActivationFunction
import shape.komputation.layers.forward.activation.softmaxVectorLayer
import shape.komputation.layers.forward.activation.tanhLayer
import shape.komputation.layers.forward.columnRepetitionLayer
import shape.komputation.layers.forward.projection.projectionLayer
import shape.komputation.layers.forward.transpositionLayer
import shape.komputation.optimization.OptimizationInstruction


class AttentiveDecoder(
    private val name : String?,
    private val numberSteps : Int,
    private val encodingDimension : Int,
    private val decodingDimension: Int,
    private val activationFunction: ActivationFunction,
    private val weightInitialization: InitializationStrategy,
    private val biasInitialization: InitializationStrategy?,
    private val optimization: OptimizationInstruction?) : CpuForwardLayerInstruction {

    override fun buildForCpu(): CpuAttentiveDecoder {

        val encodingProjectionName = concatenateNames(this.name, "encoding-projection")
        val encodingProjection = projectionLayer(encodingProjectionName, this.encodingDimension, this.encodingDimension, this.weightInitialization, null, this.optimization).buildForCpu()

        val columnRepetitionLayers = Array(this.numberSteps) { indexStep ->

            val columnRepetitionLayerName = concatenateNames(this.name, "column-repetition-$indexStep")

            columnRepetitionLayer(columnRepetitionLayerName, this.numberSteps).buildForCpu()
        }

        val attentionAdditions = Array(this.numberSteps) { indexStep ->

            val attentionAdditionName = concatenateNames(this.name, "attention-addition-$indexStep")

            additionCombination(attentionAdditionName)

        }

        val attentionPreviousStateWeightingSeriesName = concatenateNames(this.name, "attention-previous-state-weighting")
        val attentionPreviousStateWeightingStepName = concatenateNames(this.name, "attention-previous-state-weighting-step")
        val attentionPreviousStateWeighting = seriesWeighting(attentionPreviousStateWeightingSeriesName, attentionPreviousStateWeightingStepName, this.numberSteps, true, this.decodingDimension, this.encodingDimension, this.weightInitialization, this.optimization)

        val tanh = Array(this.numberSteps) { tanhLayer().buildForCpu() }

        val scoringWeightingSeriesName = concatenateNames(this.name, "scoring-weighting")
        val scoringWeightingStepName = concatenateNames(this.name, "scoring-weighting-step")
        val scoringWeighting = seriesWeighting(scoringWeightingSeriesName, scoringWeightingStepName, this.numberSteps, false, this.encodingDimension, 1, this.weightInitialization, this.optimization)

        val softmax = Array(this.numberSteps) { softmaxVectorLayer().buildForCpu() }

        val transposition = Array(this.numberSteps) { transpositionLayer().buildForCpu() }

        val attendedEncodingWeightingSeriesName = concatenateNames(this.name, "attended-encoding-weighting")
        val attendedEncodingWeightingStepName = concatenateNames(this.name, "attended-encoding-weighting-step")
        val attendedEncodingWeighting = seriesWeighting(attendedEncodingWeightingSeriesName, attendedEncodingWeightingStepName, this.numberSteps, false, this.encodingDimension, this.encodingDimension, this.weightInitialization, this.optimization)

        val decodingPreviousStateWeightingSeriesName = concatenateNames(this.name, "decoding-previous-state-weighting")
        val decodingPreviousStateWeightingStepName = concatenateNames(this.name, "decoding-previous-state-weighting-step")
        val decodingPreviousStateWeighting = seriesWeighting(decodingPreviousStateWeightingSeriesName, decodingPreviousStateWeightingStepName, this.numberSteps, true, this.decodingDimension, this.decodingDimension, this.weightInitialization, this.optimization)

        val decodingAdditions = Array(this.numberSteps) { indexStep ->

            val decodingAdditionName = concatenateNames(this.name, "decoding-addition-$indexStep")

            AdditionCombination(decodingAdditionName)

        }

        val activationName = concatenateNames(this.name, "decoding-activation")
        val activations = Array(this.numberSteps) { index ->

            cpuActivationLayer(concatenateNames(activationName, index.toString()), this.activationFunction, this.decodingDimension, this.numberSteps).buildForCpu()

        }

        val bias =
            if(this.biasInitialization == null)
                null
            else {

                val biasName =  concatenateNames(this.name, "bias")

                seriesBias(biasName, this.decodingDimension, this.biasInitialization, this.optimization)
            }

        val attentiveDecoder = CpuAttentiveDecoder(
            this.name,
            this.numberSteps,
            this.encodingDimension,
            this.decodingDimension,
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
            bias,
            activations
        )

        return attentiveDecoder

    }


}


fun attentiveDecoder(
    numberSteps : Int,
    encodingDimension : Int,
    decodingDimension: Int,
    activationFunction: ActivationFunction,
    weightInitialization: InitializationStrategy,
    biasInitialization: InitializationStrategy?,
    optimization: OptimizationInstruction?) =

    attentiveDecoder(
        null,
        numberSteps,
        encodingDimension,
        decodingDimension,
        activationFunction,
        weightInitialization,
        biasInitialization,
        optimization
    )

fun attentiveDecoder(
    name : String?,
    numberSteps : Int,
    encodingDimension : Int,
    decodingDimension: Int,
    activation: ActivationFunction,
    weightInitialization: InitializationStrategy,
    biasInitialization: InitializationStrategy?,
    optimization: OptimizationInstruction?) =

    AttentiveDecoder(
        name,
        numberSteps,
        encodingDimension,
        decodingDimension,
        activation,
        weightInitialization,
        biasInitialization,
        optimization
    )
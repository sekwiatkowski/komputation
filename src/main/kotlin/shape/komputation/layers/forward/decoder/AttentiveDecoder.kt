package shape.komputation.layers.forward.decoder

import shape.komputation.cpu.combination.AdditionCombination
import shape.komputation.cpu.combination.additionCombination
import shape.komputation.cpu.forward.activation.activationLayer
import shape.komputation.cpu.forward.decoder.CpuAttentiveDecoder
import shape.komputation.cpu.forward.projection.seriesBias
import shape.komputation.cpu.forward.projection.seriesWeighting
import shape.komputation.functions.activation.ActivationFunction
import shape.komputation.initialization.InitializationStrategy
import shape.komputation.layers.CpuForwardLayerInstruction
import shape.komputation.layers.concatenateNames
import shape.komputation.layers.forward.activation.softmaxVectorLayer
import shape.komputation.layers.forward.activation.tanhLayer
import shape.komputation.layers.forward.columnRepetitionLayer
import shape.komputation.layers.forward.projection.projectionLayer
import shape.komputation.layers.forward.transpositionLayer
import shape.komputation.optimization.OptimizationStrategy


class AttentiveDecoder(
    private val name : String?,
    private val numberSteps : Int,
    private val encodingDimension : Int,
    private val decodingDimension: Int,
    private val activationFunction: ActivationFunction,
    private val weightInitializationStrategy: InitializationStrategy,
    private val biasInitializationStrategy: InitializationStrategy?,
    private val optimizationStrategy: OptimizationStrategy) : CpuForwardLayerInstruction {

    override fun buildForCpu(): CpuAttentiveDecoder {

        val encodingProjectionName = concatenateNames(this.name, "encoding-projection")
        val encodingProjection = projectionLayer(encodingProjectionName, this.encodingDimension, this.encodingDimension, this.weightInitializationStrategy, null, this.optimizationStrategy).buildForCpu()

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
        val attentionPreviousStateWeighting = seriesWeighting(attentionPreviousStateWeightingSeriesName, attentionPreviousStateWeightingStepName, this.numberSteps, true, this.decodingDimension, this.encodingDimension, this.weightInitializationStrategy, this.optimizationStrategy)

        val tanh = Array(this.numberSteps) { tanhLayer().buildForCpu() }

        val scoringWeightingSeriesName = concatenateNames(this.name, "scoring-weighting")
        val scoringWeightingStepName = concatenateNames(this.name, "scoring-weighting-step")
        val scoringWeighting = seriesWeighting(scoringWeightingSeriesName, scoringWeightingStepName, this.numberSteps, false, this.encodingDimension, 1, this.weightInitializationStrategy, this.optimizationStrategy)

        val softmax = Array(this.numberSteps) { softmaxVectorLayer().buildForCpu() }

        val transposition = Array(this.numberSteps) { transpositionLayer().buildForCpu() }

        val attendedEncodingWeightingSeriesName = concatenateNames(this.name, "attended-encoding-weighting")
        val attendedEncodingWeightingStepName = concatenateNames(this.name, "attended-encoding-weighting-step")
        val attendedEncodingWeighting = seriesWeighting(attendedEncodingWeightingSeriesName, attendedEncodingWeightingStepName, this.numberSteps, false, this.encodingDimension, this.encodingDimension, this.weightInitializationStrategy, this.optimizationStrategy)

        val decodingPreviousStateWeightingSeriesName = concatenateNames(this.name, "decoding-previous-state-weighting")
        val decodingPreviousStateWeightingStepName = concatenateNames(this.name, "decoding-previous-state-weighting-step")
        val decodingPreviousStateWeighting = seriesWeighting(decodingPreviousStateWeightingSeriesName, decodingPreviousStateWeightingStepName, this.numberSteps, true, this.decodingDimension, this.decodingDimension, this.weightInitializationStrategy, this.optimizationStrategy)

        val decodingAdditions = Array(this.numberSteps) { indexStep ->

            val decodingAdditionName = concatenateNames(this.name, "decoding-addition-$indexStep")

            AdditionCombination(decodingAdditionName)

        }

        val activationName = concatenateNames(this.name, "decoding-activation")
        val activations = Array(this.numberSteps) { index ->

            activationLayer(concatenateNames(activationName, index.toString()), this.activationFunction).buildForCpu()

        }

        val bias =
            if(this.biasInitializationStrategy == null)
                null
            else {

                val biasName =  concatenateNames(this.name, "bias")

                seriesBias(biasName, this.decodingDimension, this.biasInitializationStrategy, this.optimizationStrategy)
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
            activations,
            bias
        )

        return attentiveDecoder

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
    optimizationStrategy: OptimizationStrategy) =

    AttentiveDecoder(
        name,
        numberSteps,
        encodingDimension,
        decodingDimension,
        activationFunction,
        weightInitializationStrategy,
        biasInitializationStrategy,
        optimizationStrategy
    )
package shape.komputation.demos.runningtotal

import shape.komputation.functions.activation.ActivationFunction
import shape.komputation.initialization.createGaussianInitializer
import shape.komputation.initialization.createIdentityInitializer
import shape.komputation.initialization.createZeroInitializer
import shape.komputation.layers.entry.InputLayer
import shape.komputation.layers.feedforward.decoder.createDecoderUnit
import shape.komputation.layers.feedforward.decoder.createMultiInputDecoder
import shape.komputation.layers.feedforward.encoder.MultiOutputEncoder
import shape.komputation.layers.feedforward.encoder.createMultiOutputEncoder
import shape.komputation.layers.feedforward.units.createSimpleRecurrentUnit
import shape.komputation.loss.SquaredLoss
import shape.komputation.matrix.DoubleMatrix
import shape.komputation.matrix.Matrix
import shape.komputation.matrix.SequenceMatrix
import shape.komputation.networks.Network
import shape.komputation.networks.printLoss
import shape.komputation.optimization.stochasticGradientDescent
import java.util.*

fun main(args: Array<String>) {

    val numberSteps = 4
    val exclusiveUpperLimit = 10
    val hiddenDimension = 4
    val numberExamples = Math.pow(exclusiveUpperLimit.toDouble(), numberSteps.toDouble()).toInt()
    val numberIterations = 10
    val batchSize = 4

    val random = Random(1)

    val encoderPreviousStateWeightInitializationStrategy = createIdentityInitializer()
    val encoderInputWeightInitializationStrategy = createGaussianInitializer(random, 0.0, 0.001)
    val encoderBiasInitializationStrategy = createZeroInitializer()

    val decoderPreviousStateWeightInitializationStrategy = createIdentityInitializer()
    val decoderInputWeightInitializationStrategy = createGaussianInitializer(random, 0.0, 0.001)
    val decoderBiasInitializationStrategy = createZeroInitializer()
    val decoderOutputWeightInitializationStrategy = createGaussianInitializer(random, 0.0, 0.001)

    val optimizationStrategy = stochasticGradientDescent(0.001)

    val inputs = Array<Matrix>(numberExamples) {

        SequenceMatrix(numberSteps, 1, 1, DoubleArray(numberSteps) { random.nextInt(exclusiveUpperLimit).toDouble() })

    }

    val targets = Array<DoubleMatrix>(numberExamples) { indexExample ->

        val input = inputs[indexExample]

        input as SequenceMatrix

        val targetEntries = input
            .entries
            .foldIndexed(arrayListOf<Double>()) { index, list, current ->

                list.add(list.getOrElse(index-1, { 0.0 }) + current)

                list

            }
            .toDoubleArray()

        SequenceMatrix(numberSteps, 1, 1, targetEntries)

    }

    val encoderUnit = createSimpleRecurrentUnit(
        numberSteps,
        1,
        hiddenDimension,
        encoderInputWeightInitializationStrategy,
        encoderPreviousStateWeightInitializationStrategy,
        encoderBiasInitializationStrategy,
        ActivationFunction.Identity,
        optimizationStrategy)

    val encoder = createMultiOutputEncoder(
        encoderUnit,
        numberSteps,
        1,
        hiddenDimension)

    val decoderUnit = createDecoderUnit(
        numberSteps,
        hiddenDimension,
        hiddenDimension,
        1,
        false,
        decoderInputWeightInitializationStrategy,
        decoderPreviousStateWeightInitializationStrategy,
        decoderBiasInitializationStrategy,
        ActivationFunction.Identity,
        decoderOutputWeightInitializationStrategy,
        ActivationFunction.Identity,
        optimizationStrategy
    )

    val decoder = createMultiInputDecoder(
        decoderUnit,
        numberSteps,
        hiddenDimension,
        hiddenDimension,
        1

    )

    val network = Network(
        InputLayer(),
        encoder,
        decoder
    )

    network.train(
        inputs,
        targets,
        SquaredLoss(),
        numberIterations,
        batchSize,
        printLoss
    )

}

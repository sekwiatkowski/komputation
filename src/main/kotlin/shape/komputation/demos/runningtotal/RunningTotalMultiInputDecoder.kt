package shape.komputation.demos.runningtotal

import shape.komputation.functions.activation.ActivationFunction
import shape.komputation.initialization.createGaussianInitializer
import shape.komputation.initialization.createIdentityInitializer
import shape.komputation.initialization.createZeroInitializer
import shape.komputation.layers.entry.InputLayer
import shape.komputation.layers.feedforward.decoder.createMultiInputDecoder
import shape.komputation.layers.feedforward.encoder.createMultiOutputEncoder
import shape.komputation.layers.feedforward.units.createSimpleRecurrentUnit
import shape.komputation.loss.SquaredLoss
import shape.komputation.networks.Network
import shape.komputation.networks.printLoss
import shape.komputation.optimization.stochasticGradientDescent
import java.util.*

fun main(args: Array<String>) {

    val exclusiveUpperLimit = 10
    val numberSteps = 4
    val numberExamples = Math.pow(exclusiveUpperLimit.toDouble(), numberSteps.toDouble()).toInt()
    val hiddenDimension = 4
    val numberIterations = 30
    val batchSize = 4

    val random = Random(1)

    val identityInitializationStrategy = createIdentityInitializer()
    val gaussianInitializationStrategy = createGaussianInitializer(random, 0.0, 0.001)
    val zeroInitializationStrategy = createZeroInitializer()

    val optimizationStrategy = stochasticGradientDescent(0.001)

    val inputs = RunningTotalData.generateInputs(random, numberExamples, numberSteps, exclusiveUpperLimit)

    val targets = RunningTotalData.generateTargets(inputs, numberSteps)

    val encoderUnit = createSimpleRecurrentUnit(
        numberSteps,
        hiddenDimension,
        1,
        gaussianInitializationStrategy,
        identityInitializationStrategy,
        zeroInitializationStrategy,
        ActivationFunction.Identity,
        optimizationStrategy)

    val encoder = createMultiOutputEncoder(
        encoderUnit,
        numberSteps,
        1,
        hiddenDimension)

    val decoderUnit = createSimpleRecurrentUnit(
        numberSteps,
        hiddenDimension,
        hiddenDimension,
        identityInitializationStrategy,
        gaussianInitializationStrategy,
        zeroInitializationStrategy,
        ActivationFunction.Identity,
        optimizationStrategy
    )

    val decoder = createMultiInputDecoder(
        numberSteps,
        hiddenDimension,
        hiddenDimension,
        1,
        decoderUnit,
        gaussianInitializationStrategy,
        zeroInitializationStrategy,
        ActivationFunction.Identity,
        optimizationStrategy
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
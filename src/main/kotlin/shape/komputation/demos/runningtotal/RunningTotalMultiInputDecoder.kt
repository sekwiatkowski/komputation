package shape.komputation.demos.runningtotal

import shape.komputation.functions.activation.ActivationFunction
import shape.komputation.initialization.gaussianInitialization
import shape.komputation.initialization.identityInitialization
import shape.komputation.initialization.zeroInitialization
import shape.komputation.layers.entry.inputLayer
import shape.komputation.layers.forward.decoder.multiInputDecoder
import shape.komputation.layers.forward.encoder.multiOutputEncoder
import shape.komputation.layers.forward.units.simpleRecurrentUnit
import shape.komputation.loss.squaredLoss
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

    val identityInitializationStrategy = identityInitialization()
    val gaussianInitializationStrategy = gaussianInitialization(random, 0.0, 0.001)
    val zeroInitializationStrategy = zeroInitialization()

    val optimizationStrategy = stochasticGradientDescent(0.001)

    val inputs = RunningTotalData.generateInputs(random, numberExamples, numberSteps, exclusiveUpperLimit)

    val targets = RunningTotalData.generateTargets(inputs, numberSteps)

    val encoderUnit = simpleRecurrentUnit(
        numberSteps,
        hiddenDimension,
        1,
        gaussianInitializationStrategy,
        identityInitializationStrategy,
        zeroInitializationStrategy,
        ActivationFunction.Identity,
        optimizationStrategy)

    val encoder = multiOutputEncoder(
        encoderUnit,
        numberSteps,
        1,
        hiddenDimension)

    val decoderUnit = simpleRecurrentUnit(
        numberSteps,
        hiddenDimension,
        hiddenDimension,
        identityInitializationStrategy,
        gaussianInitializationStrategy,
        zeroInitializationStrategy,
        ActivationFunction.Identity,
        optimizationStrategy
    )

    val decoder = multiInputDecoder(
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
        inputLayer(),
        encoder,
        decoder
    )

    network.train(
        inputs,
        targets,
        squaredLoss(),
        numberIterations,
        batchSize,
        printLoss
    )

}
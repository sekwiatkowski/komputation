package shape.komputation.cpu.demos.runningtotal

import shape.komputation.cpu.Network
import shape.komputation.cpu.layers.forward.units.simpleRecurrentUnit
import shape.komputation.cpu.printLoss
import shape.komputation.demos.runningtotal.RunningTotalData
import shape.komputation.initialization.gaussianInitialization
import shape.komputation.initialization.identityInitialization
import shape.komputation.initialization.zeroInitialization
import shape.komputation.layers.entry.inputLayer
import shape.komputation.layers.forward.activation.ActivationFunction
import shape.komputation.layers.forward.decoder.multiInputDecoder
import shape.komputation.layers.forward.encoder.multiOutputEncoder
import shape.komputation.loss.squaredLoss
import shape.komputation.matrix.IntMath
import shape.komputation.optimization.stochasticGradientDescent
import java.util.*

fun main(args: Array<String>) {

    val exclusiveUpperLimit = 10
    val numberSteps = 4
    val numberExamples = IntMath.pow(exclusiveUpperLimit, numberSteps)
    val hiddenDimension = 4
    val numberIterations = 30
    val batchSize = 4

    val random = Random(1)

    val identityInitializationStrategy = identityInitialization()
    val gaussianInitializationStrategy = gaussianInitialization(random, 0.0f, 0.001f)
    val zeroInitializationStrategy = zeroInitialization()

    val optimizationStrategy = stochasticGradientDescent(0.001f)

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
        inputLayer(numberSteps),
        encoder,
        decoder
    )

    network.train(
        inputs,
        targets,
        squaredLoss(1),
        numberIterations,
        batchSize,
        printLoss
    )

}
package shape.komputation.demos.addition

import shape.komputation.cpu.layers.forward.units.minimalGatedUnit
import shape.komputation.cpu.loss.squaredLoss
import shape.komputation.initialization.gaussianInitialization
import shape.komputation.initialization.identityInitialization
import shape.komputation.initialization.zeroInitialization
import shape.komputation.layers.entry.inputLayer
import shape.komputation.layers.forward.encoder.singleOutputEncoder
import shape.komputation.layers.forward.projection.projectionLayer
import shape.komputation.networks.Network
import shape.komputation.networks.printLoss
import shape.komputation.optimization.stochasticGradientDescent
import java.util.*

fun main(args: Array<String>) {

    val random = Random(1)
    val length = 40
    val numberExamples = 10_000
    val batchSize = 1
    val inputDimension = 2
    val hiddenDimension = 20
    val numberIterations = 100

    val inputs = AdditionProblemData.generateInputs(numberExamples, random, length)

    val targets = AdditionProblemData.generateTarget(inputs)

    val identityInitializationStrategy = identityInitialization()
    val gaussianInitializationStrategy = gaussianInitialization(random, 0.0, 0.001)
    val zeroInitializationStrategy = zeroInitialization()

    val optimizationStrategy = stochasticGradientDescent(0.001)

    val encoderUnit = minimalGatedUnit(
        length,
        hiddenDimension,
        inputDimension,
        identityInitializationStrategy,
        gaussianInitializationStrategy,
        zeroInitializationStrategy,
        identityInitializationStrategy,
        gaussianInitializationStrategy,
        zeroInitializationStrategy,
        optimizationStrategy
    )

    val encoder = singleOutputEncoder(encoderUnit, length, inputDimension, hiddenDimension)

    val outputProjection = projectionLayer(hiddenDimension, 1, gaussianInitializationStrategy, gaussianInitializationStrategy, optimizationStrategy)

    val network = Network(
        inputLayer(),
        encoder,
        outputProjection
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
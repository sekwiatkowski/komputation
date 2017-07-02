package shape.komputation.demos.addition

import shape.komputation.initialization.createGaussianInitializer
import shape.komputation.initialization.createIdentityInitializer
import shape.komputation.initialization.createZeroInitializer
import shape.komputation.layers.entry.InputLayer
import shape.komputation.layers.forward.encoder.createSingleOutputEncoder
import shape.komputation.layers.forward.projection.createProjectionLayer
import shape.komputation.layers.forward.units.createMinimalGatedUnit
import shape.komputation.loss.SquaredLoss
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

    val identityInitializationStrategy = createIdentityInitializer()
    val gaussianInitializationStrategy = createGaussianInitializer(random, 0.0, 0.001)
    val zeroInitializationStrategy = createZeroInitializer()

    val optimizationStrategy = stochasticGradientDescent(0.001)

    val encoderUnit = createMinimalGatedUnit(
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

    val encoder = createSingleOutputEncoder(encoderUnit, length, inputDimension, hiddenDimension)

    val outputProjection = createProjectionLayer(hiddenDimension, 1, gaussianInitializationStrategy, gaussianInitializationStrategy, optimizationStrategy)

    val network = Network(
        InputLayer(),
        encoder,
        outputProjection
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
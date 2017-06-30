package shape.komputation.demos.addition

import shape.komputation.functions.activation.ActivationFunction
import shape.komputation.initialization.createGaussianInitializer
import shape.komputation.initialization.createIdentityInitializer
import shape.komputation.initialization.createZeroInitializer
import shape.komputation.layers.entry.InputLayer
import shape.komputation.layers.feedforward.encoder.createSingleOutputEncoder
import shape.komputation.layers.feedforward.projection.createProjectionLayer
import shape.komputation.layers.feedforward.units.createSimpleRecurrentUnit
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

    val previousStateWeightInitializationStrategy = createIdentityInitializer()
    val inputWeightInitializationStrategy = createGaussianInitializer(random, 0.0, 0.001)
    val biasInitializationStrategy = createZeroInitializer()

    val optimizationStrategy = stochasticGradientDescent(0.001)

    val encoderUnit = createSimpleRecurrentUnit(
        length,
        hiddenDimension,
        inputDimension,
        previousStateWeightInitializationStrategy,
        inputWeightInitializationStrategy,
        biasInitializationStrategy,
        ActivationFunction.ReLU,
        optimizationStrategy
    )

    val encoder = createSingleOutputEncoder(encoderUnit, length, inputDimension, hiddenDimension)

    val projectionWeightInitializationStrategy = createGaussianInitializer(random, 0.0, 0.001)

    val outputProjection = createProjectionLayer(hiddenDimension, 1, true, projectionWeightInitializationStrategy, optimizationStrategy)

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
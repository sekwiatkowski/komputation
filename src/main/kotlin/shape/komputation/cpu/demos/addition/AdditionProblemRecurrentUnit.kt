package shape.komputation.cpu.demos.addition

import shape.komputation.cpu.Network
import shape.komputation.cpu.layers.forward.units.simpleRecurrentUnit
import shape.komputation.cpu.printLoss
import shape.komputation.demos.addition.AdditionProblemData
import shape.komputation.initialization.gaussianInitialization
import shape.komputation.initialization.identityInitialization
import shape.komputation.initialization.zeroInitialization
import shape.komputation.layers.entry.inputLayer
import shape.komputation.layers.forward.activation.ActivationFunction
import shape.komputation.layers.forward.encoder.singleOutputEncoder
import shape.komputation.layers.forward.projection.projectionLayer
import shape.komputation.loss.squaredLoss
import shape.komputation.optimization.stochasticGradientDescent
import java.util.*

fun main(args: Array<String>) {

    val random = Random(1)
    val length = 40
    val numberExamples = 10_000
    val batchSize = 1
    val inputDimension = 2
    val hiddenDimension = 20
    val outputDimension = 1
    val numberIterations = 100

    val inputs = AdditionProblemData.generateInputs(numberExamples, random, length)

    val targets = AdditionProblemData.generateTarget(inputs)

    val identityInitializationStrategy = identityInitialization()
    val gaussianInitializationStrategy = gaussianInitialization(random, 0.0f, 0.001f)
    val zeroInitializationStrategy = zeroInitialization()

    val optimizationStrategy = stochasticGradientDescent(0.001f)

    val encoderUnit = simpleRecurrentUnit(
        length,
        inputDimension,
        hiddenDimension,
        identityInitializationStrategy,
        gaussianInitializationStrategy,
        zeroInitializationStrategy,
        ActivationFunction.ReLU,
        optimizationStrategy
    )

    val network = Network(
        inputLayer(inputDimension),
        singleOutputEncoder(encoderUnit, length, inputDimension, hiddenDimension),
        projectionLayer(hiddenDimension, outputDimension, gaussianInitializationStrategy, gaussianInitializationStrategy, optimizationStrategy)
    )

    network.train(
        inputs,
        targets,
        squaredLoss(outputDimension),
        numberIterations,
        batchSize,
        printLoss
    )

}
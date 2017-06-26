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
import shape.komputation.matrix.*
import shape.komputation.networks.Network
import shape.komputation.networks.printLoss
import shape.komputation.optimization.stochasticGradientDescent
import java.util.*

fun main(args: Array<String>) {

    val random = Random(1)
    val length = 8
    val numberExamples = 100_000
    val batchSize = 4
    val inputDimension = 2
    val hiddenDimension = 5
    val numberIterations = 10

    val inputs = Array<Matrix>(numberExamples) { generateInput(random, length) }

    val targets = Array(numberExamples) { indexInput ->

        calculateSolution(inputs[indexInput] as SequenceMatrix)

    }

    val previousStateProjectionInitializationStrategy = createIdentityInitializer()
    val inputProjectionInitializationStrategy = createGaussianInitializer(random, 0.0, 0.001)
    val biasInitializationStrategy = createZeroInitializer()

    val projectionWeightInitializationStrategy = createGaussianInitializer(random, 0.0, 0.001)

    val optimizationStrategy = stochasticGradientDescent(0.01)

    val encoderUnit = createSimpleRecurrentUnit(
        length,
        inputDimension,
        hiddenDimension,
        inputProjectionInitializationStrategy,
        previousStateProjectionInitializationStrategy,
        biasInitializationStrategy,
        ActivationFunction.ReLU,
        optimizationStrategy
    )

    val encoder = createSingleOutputEncoder(encoderUnit, length, inputDimension, hiddenDimension)

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

private fun generateInput(random: Random, length: Int): SequenceMatrix {

    val input = sequence(length, 2) {

        doubleArrayOf(
            random.nextDouble(),
            0.0)

    }

    val firstStep = random.nextInt(length)
    val secondStep = random.nextInt(length).let { candidate ->

        if (candidate == firstStep) {

            if (firstStep == length - 1) {
                firstStep - 1
            }
            else {
                firstStep + 1
            }

        }
        else {

            candidate
        }

    }

    input.setEntry(firstStep, 1, 0, 1.0)
    input.setEntry(secondStep, 1, 0, 1.0)

    return input

}

private fun calculateSolution(input: SequenceMatrix): DoubleMatrix {

    var solution = 0.0
    for (indexStep in 0..input.numberSteps - 1) {

        val step = input.getStep(indexStep)

        solution += step.entries[0] * step.entries[1]

    }

    return doubleScalar(solution)

}
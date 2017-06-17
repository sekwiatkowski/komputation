package shape.komputation.demos

import shape.komputation.initialization.createGaussianInitializer
import shape.komputation.initialization.createIdentityInitializer
import shape.komputation.initialization.createZeroInitializer
import shape.komputation.layers.entry.InputLayer
import shape.komputation.layers.feedforward.activation.ReluLayer
import shape.komputation.layers.feedforward.projection.createProjectionLayer
import shape.komputation.layers.recurrent.createRecurrentLayer
import shape.komputation.loss.SquaredLoss
import shape.komputation.matrix.DoubleMatrix
import shape.komputation.matrix.Matrix
import shape.komputation.matrix.doubleRowVector
import shape.komputation.matrix.doubleScalar
import shape.komputation.networks.RecurrentNetwork
import shape.komputation.networks.printLoss
import shape.komputation.optimization.stochasticGradientDescent
import java.util.*

fun main(args: Array<String>) {

    val random = Random(1)
    val length = 10
    val numberExamples = 100_000
    val batchSize = 4
    val inputDimension = 2
    val hiddenDimension = 5

    val inputs = Array(numberExamples) { generateInput(random, length) }

    val targets = Array(numberExamples) { indexInput ->

        calculateSolution(inputs[indexInput])

    }

    val stateWeightInitializationStrategy = createIdentityInitializer()
    val inputWeightInitializationStrategy = createGaussianInitializer(random, 0.0, 0.001)
    val biasInitializationStrategy = createZeroInitializer()

    val projectionWeightInitializationStrategy = createGaussianInitializer(random, 0.0, 0.001)

    val optimizationStrategy = stochasticGradientDescent(0.01)

    val recurrentLayer = createRecurrentLayer(
        inputDimension,
        hiddenDimension,
        ReluLayer(),
        stateWeightInitializationStrategy,
        inputWeightInitializationStrategy,
        biasInitializationStrategy,
        optimizationStrategy
    )

    val network = RecurrentNetwork(
        InputLayer(),
        recurrentLayer,
        createProjectionLayer(hiddenDimension, 1, true, projectionWeightInitializationStrategy, optimizationStrategy)
    )

    network.train(
        inputs,
        targets,
        SquaredLoss(),
        100,
        batchSize,
        printLoss
    )

}

private fun generateInput(random: Random, length: Int): Array<Matrix> {

    val input = Array<Matrix>(length) {

        doubleRowVector(
            random.nextDouble(),
            0.0)

    }

    val firstIndex = random.nextInt(length)
    val secondIndex = random.nextInt(length).let { candidate ->

        if (candidate == firstIndex) {

            if (firstIndex == length - 1) {
                firstIndex - 1
            }
            else {
                firstIndex + 1
            }

        }
        else {

            candidate
        }

    }

    val firstSelection = input[firstIndex] as DoubleMatrix
    firstSelection.entries[1] = 1.0

    val secondSelection = input[secondIndex] as DoubleMatrix
    secondSelection.entries[1] = 1.0

    return input

}

private fun calculateSolution(input: Array<Matrix>): DoubleMatrix {

    val solution = input
        .sumByDouble { matrix ->

            matrix as DoubleMatrix

            matrix.entries[0] * matrix.entries[1]

        }

    return doubleScalar(solution)

}
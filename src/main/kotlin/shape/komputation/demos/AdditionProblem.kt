package shape.komputation.demos

import shape.komputation.initialization.createGaussianInitializer
import shape.komputation.layers.feedforward.activation.ReluLayer
import shape.komputation.layers.recurrent.createVanillaRecurrentLayer
import shape.komputation.layers.entry.InputLayer
import shape.komputation.layers.feedforward.createProjectionLayer
import shape.komputation.loss.SquaredLoss
import shape.komputation.matrix.Matrix
import shape.komputation.matrix.RealMatrix
import shape.komputation.matrix.createRealVector
import shape.komputation.network.RecurrentNetwork
import shape.komputation.network.printLoss
import shape.komputation.optimization.stochasticGradientDescent
import java.util.*

fun main(args: Array<String>) {

    val random = Random(1)
    val length = 10
    val numberExamples = 100_000
    val inputDimension = 2
    val hiddenDimension = 100

    val inputs = Array(numberExamples) { generateInput(random, length) }

    val targets = Array(numberExamples) { indexInput ->

        calculateSolution(inputs[indexInput])

    }

    val initializationStrategy = createGaussianInitializer(random, 0.0, 0.001)
    val optimizationStrategy = stochasticGradientDescent(0.01)

    val recurrentLayer = createVanillaRecurrentLayer(
        length,
        inputDimension,
        hiddenDimension,
        ReluLayer(),
        initializationStrategy,
        optimizationStrategy
    )

    val network = RecurrentNetwork(
        length,
        InputLayer(),
        recurrentLayer,
        createProjectionLayer(hiddenDimension, 1, initializationStrategy, optimizationStrategy)
    )

    network.train(
        inputs,
        targets,
        SquaredLoss(),
        10_000,
        printLoss
    )

}

private fun generateInput(random: Random, length: Int): Array<Matrix> {

    val input = Array<Matrix>(length) {

        createRealVector(
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

    (input[firstIndex] as RealMatrix).set(1, 0, 1.0)
    (input[secondIndex] as RealMatrix).set(1, 0, 1.0)

    return input

}

private fun calculateSolution(input: Array<Matrix>): RealMatrix {

    val solution = input
        .sumByDouble { matrix ->

            matrix as RealMatrix

            matrix.get(0, 0) * matrix.get(1, 0)

        }

    return createRealVector(solution)

}
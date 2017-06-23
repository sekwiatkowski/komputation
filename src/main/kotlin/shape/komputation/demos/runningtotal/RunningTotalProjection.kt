package shape.komputation.demos.runningtotal

import shape.komputation.functions.activation.ActivationFunction
import shape.komputation.initialization.createGaussianInitializer
import shape.komputation.initialization.createIdentityInitializer
import shape.komputation.initialization.createZeroInitializer
import shape.komputation.layers.entry.InputLayer
import shape.komputation.layers.feedforward.encoder.createMultiOutputEncoder
import shape.komputation.layers.feedforward.projection.createProjectionLayer
import shape.komputation.loss.SquaredLoss
import shape.komputation.matrix.DoubleMatrix
import shape.komputation.matrix.Matrix
import shape.komputation.matrix.SequenceMatrix
import shape.komputation.networks.Network
import shape.komputation.networks.printLoss
import shape.komputation.optimization.stochasticGradientDescent
import java.util.*

fun main(args: Array<String>) {

    val numberSteps = 4
    val exclusiveUpperLimit = 10
    val hiddenDimension = 4
    val numberExamples = Math.pow(exclusiveUpperLimit.toDouble(), numberSteps.toDouble()).toInt()
    val numberIterations = 100
    val batchSize = 4

    val random = Random(1)

    val previousStateWeightInitializationStrategy = createIdentityInitializer()
    val inputWeightInitializationStrategy = createGaussianInitializer(random, 0.0, 0.001)
    val biasInitializationStrategy = createZeroInitializer()

    val optimizationStrategy = stochasticGradientDescent(0.001)

    val inputs = Array<Matrix>(numberExamples) {

        SequenceMatrix(numberSteps, 1, 1, DoubleArray(numberSteps) { random.nextInt(exclusiveUpperLimit).toDouble() })

    }

    val targets = Array<DoubleMatrix>(numberExamples) { indexExample ->

        val input = inputs[indexExample]

        input as SequenceMatrix

        val targetEntries = input
            .entries
            .foldIndexed(arrayListOf<Double>()) { index, list, current ->

                list.add(list.getOrElse(index-1, { 0.0 }) + current)

                list

            }
            .toDoubleArray()

        SequenceMatrix(numberSteps, 1, 1, targetEntries)

    }

    val network = Network(
        InputLayer(),
        createMultiOutputEncoder(numberSteps, 1, hiddenDimension, inputWeightInitializationStrategy, previousStateWeightInitializationStrategy, biasInitializationStrategy, ActivationFunction.Identity, optimizationStrategy),
        createProjectionLayer(hiddenDimension, 1, false, inputWeightInitializationStrategy, optimizationStrategy)
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

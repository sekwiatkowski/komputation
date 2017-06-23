package shape.komputation.demos.reverse

import shape.komputation.functions.activation.ActivationFunction
import shape.komputation.initialization.createGaussianInitializer
import shape.komputation.initialization.createIdentityInitializer
import shape.komputation.initialization.createZeroInitializer
import shape.komputation.layers.entry.InputLayer
import shape.komputation.layers.feedforward.decoder.createSingleInputDecoder
import shape.komputation.layers.feedforward.encoder.createSingleOutputEncoder
import shape.komputation.loss.LogisticLoss
import shape.komputation.matrix.*
import shape.komputation.networks.Network
import shape.komputation.networks.printLoss
import shape.komputation.optimization.stochasticGradientDescent
import java.util.*

fun main(args: Array<String>) {

    val random = Random(1)
    val seriesLength = 5
    val numberCategories = 10
    val numberExamples = Math.pow(10.toDouble(), seriesLength.toDouble()).toInt()
    val hiddenDimension = 10
    val numberIterations = 50
    val batchSize = 4

    val inputs = Array<Matrix>(numberExamples) {

        val sequenceMatrix = zeroSequenceMatrix(seriesLength, numberCategories, 1)

        for (indexStep in 0..seriesLength - 1) {

            sequenceMatrix.setStep(indexStep, oneHotArray(numberCategories, random.nextInt(10), 1.0, 0.001))

        }

        sequenceMatrix

    }

    val targets = Array<DoubleMatrix>(numberExamples) { index ->

        val sequenceMatrix = inputs[index] as SequenceMatrix

        val reversedSequenceMatrix = zeroSequenceMatrix(seriesLength, numberCategories, 1)

        for (indexStep in 0..seriesLength - 1) {

            val reverseStep = seriesLength - indexStep - 1

            val originalStep = sequenceMatrix.getStep(reverseStep).entries

            reversedSequenceMatrix.setStep(indexStep, originalStep)
        }

        reversedSequenceMatrix

    }

    val previousStateWeightInitializationStrategy = createIdentityInitializer()
    val inputWeightInitializationStrategy = createGaussianInitializer(random, 0.0, 0.001)
    val biasInitializationStrategy = createZeroInitializer()

    val previousOutputWeightInitializationStrategy = createGaussianInitializer(random, 0.0, 0.001)
    val outputWeightInitializationStrategy = createZeroInitializer()

    val optimizationStrategy = stochasticGradientDescent(0.001)

    val stateActivationFunction = ActivationFunction.ReLU
    val outputActivationFunction = ActivationFunction.Softmax

    val encoder = createSingleOutputEncoder(seriesLength, numberCategories, hiddenDimension, inputWeightInitializationStrategy, previousStateWeightInitializationStrategy, biasInitializationStrategy, stateActivationFunction, optimizationStrategy)
    val decoder = createSingleInputDecoder(seriesLength, numberCategories, hiddenDimension, numberCategories, previousOutputWeightInitializationStrategy, previousStateWeightInitializationStrategy, null, stateActivationFunction, outputWeightInitializationStrategy, outputActivationFunction, optimizationStrategy)

    val network = Network(
        InputLayer(),
        encoder,
        decoder
    )

    network.train(
        inputs,
        targets,
        LogisticLoss(),
        numberIterations,
        batchSize,
        printLoss
    )

}

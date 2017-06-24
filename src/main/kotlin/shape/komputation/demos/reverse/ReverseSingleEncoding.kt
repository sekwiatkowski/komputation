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
    val batchSize = 10

    val inputs = Array<Matrix>(numberExamples) {

        val sequenceMatrix = zeroSequenceMatrix(seriesLength, numberCategories, 1)

        for (indexStep in 0..seriesLength - 1) {

            sequenceMatrix.setStep(indexStep, oneHotArray(numberCategories, random.nextInt(10), 1.0))

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

    val identityInitialization = createIdentityInitializer()
    val gaussianInitialization = createGaussianInitializer(random, 0.0, 0.001)
    val zeroInitialization = createZeroInitializer()

    val optimizationStrategy = stochasticGradientDescent(0.001)

    val encoder = createSingleOutputEncoder(seriesLength, numberCategories, hiddenDimension, gaussianInitialization, identityInitialization, zeroInitialization, ActivationFunction.Tanh, optimizationStrategy)
    val decoder = createSingleInputDecoder(seriesLength, numberCategories, hiddenDimension, numberCategories, gaussianInitialization, identityInitialization, null, ActivationFunction.Tanh, gaussianInitialization, ActivationFunction.Softmax, optimizationStrategy)

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

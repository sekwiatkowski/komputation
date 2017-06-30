package shape.komputation.demos.reverse

import shape.komputation.functions.activation.ActivationFunction
import shape.komputation.initialization.createGaussianInitializer
import shape.komputation.initialization.createIdentityInitializer
import shape.komputation.layers.entry.InputLayer
import shape.komputation.layers.feedforward.decoder.createAttentiveDecoder
import shape.komputation.layers.feedforward.encoder.createMultiOutputEncoder
import shape.komputation.layers.feedforward.units.createSimpleRecurrentUnit
import shape.komputation.loss.LogisticLoss
import shape.komputation.networks.Network
import shape.komputation.networks.printLoss
import shape.komputation.optimization.stochasticGradientDescent
import java.util.*

fun main(args: Array<String>) {

    val random = Random(1)
    val seriesLength = 6
    val numberCategories = 10
    val numberExamples = Math.pow(10.toDouble(), seriesLength.toDouble()).toInt()
    val hiddenDimension = 10
    val numberIterations = 10
    val batchSize = 1

    val inputs = ReverseData.generateInputs(random, numberExamples, seriesLength, numberCategories)
    val targets = ReverseData.generateTargets(inputs, seriesLength, numberCategories)

    val identityInitializationStrategy = createIdentityInitializer()
    val gaussianInitializationStrategy = createGaussianInitializer(random, 0.0, 0.001)

    val optimizationStrategy = stochasticGradientDescent(0.001)

    val encoderUnit = createSimpleRecurrentUnit(
        seriesLength,
        hiddenDimension,
        numberCategories,
        gaussianInitializationStrategy,
        identityInitializationStrategy,
        gaussianInitializationStrategy,
        ActivationFunction.ReLU,
        optimizationStrategy)

    val encoder = createMultiOutputEncoder(
        encoderUnit,
        seriesLength,
        numberCategories,
        hiddenDimension
    )

    val decoder = createAttentiveDecoder(
        seriesLength,
        hiddenDimension,
        hiddenDimension,
        ActivationFunction.Sigmoid,
        gaussianInitializationStrategy,
        gaussianInitializationStrategy,
        optimizationStrategy)

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

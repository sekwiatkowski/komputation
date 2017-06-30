package shape.komputation.demos.reverse

import shape.komputation.functions.activation.ActivationFunction
import shape.komputation.initialization.createGaussianInitializer
import shape.komputation.initialization.createIdentityInitializer
import shape.komputation.initialization.createZeroInitializer
import shape.komputation.layers.entry.InputLayer
import shape.komputation.layers.feedforward.createConcatenation
import shape.komputation.layers.feedforward.decoder.createSingleInputDecoder
import shape.komputation.layers.feedforward.encoder.createSingleOutputEncoder
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
    val numberExamples = 10_000
    val hiddenDimension = 30
    val numberIterations = 100
    val batchSize = 1

    val inputs = ReverseData.generateInputs(random, numberExamples, seriesLength, numberCategories)
    val targets = ReverseData.generateTargets(inputs, seriesLength, numberCategories)

    val identityInitialization = createIdentityInitializer()
    val gaussianInitialization = createGaussianInitializer(random, 0.0, 0.001)
    val zeroInitialization = createZeroInitializer()

    val optimizationStrategy = stochasticGradientDescent(0.001)

    val forwardEncoderUnit = createSimpleRecurrentUnit(
        seriesLength,
        hiddenDimension,
        numberCategories,
        gaussianInitialization,
        identityInitialization,
        zeroInitialization,
        ActivationFunction.ReLU,
        optimizationStrategy
    )

    val backwardEncoderUnit = createSimpleRecurrentUnit(
        seriesLength,
        hiddenDimension,
        numberCategories,
        gaussianInitialization,
        identityInitialization,
        zeroInitialization,
        ActivationFunction.ReLU,
        optimizationStrategy
    )

    val decoderUnit = createSimpleRecurrentUnit(
        seriesLength,
        2 * hiddenDimension,
        numberCategories,
        identityInitialization,
        gaussianInitialization,
        zeroInitialization,
        ActivationFunction.ReLU,
        optimizationStrategy
    )

    val network = Network(
        InputLayer(),
        createConcatenation(
            createSingleOutputEncoder(forwardEncoderUnit, seriesLength, numberCategories, hiddenDimension, false),
            createSingleOutputEncoder(backwardEncoderUnit, seriesLength, numberCategories, hiddenDimension, true)
        ),
        createSingleInputDecoder(seriesLength, 2 * hiddenDimension, numberCategories, decoderUnit, gaussianInitialization, null, ActivationFunction.Softmax, optimizationStrategy)
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

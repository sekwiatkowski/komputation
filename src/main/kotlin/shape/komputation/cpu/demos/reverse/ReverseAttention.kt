package shape.komputation.cpu.demos.reverse

import shape.komputation.cpu.Network
import shape.komputation.cpu.layers.forward.units.simpleRecurrentUnit
import shape.komputation.cpu.printLoss
import shape.komputation.demos.reverse.ReverseData
import shape.komputation.initialization.gaussianInitialization
import shape.komputation.initialization.identityInitialization
import shape.komputation.layers.entry.inputLayer
import shape.komputation.layers.forward.activation.ActivationFunction
import shape.komputation.layers.forward.decoder.attentiveDecoder
import shape.komputation.layers.forward.encoder.multiOutputEncoder
import shape.komputation.loss.logisticLoss
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

    val identityInitializationStrategy = identityInitialization()
    val gaussianInitializationStrategy = gaussianInitialization(random, 0.0, 0.001)

    val optimizationStrategy = stochasticGradientDescent(0.001)

    val encoderUnit = simpleRecurrentUnit(
        seriesLength,
        hiddenDimension,
        numberCategories,
        gaussianInitializationStrategy,
        identityInitializationStrategy,
        gaussianInitializationStrategy,
        ActivationFunction.ReLU,
        optimizationStrategy)

    val encoder = multiOutputEncoder(
        encoderUnit,
        seriesLength,
        numberCategories,
        hiddenDimension
    )

    val decoder = attentiveDecoder(
        seriesLength,
        hiddenDimension,
        hiddenDimension,
        ActivationFunction.Sigmoid,
        gaussianInitializationStrategy,
        gaussianInitializationStrategy,
        optimizationStrategy)

    val network = Network(
        inputLayer(seriesLength),
        encoder,
        decoder
    )

    network.train(
        inputs,
        targets,
        logisticLoss(numberCategories),
        numberIterations,
        batchSize,
        printLoss
    )

}

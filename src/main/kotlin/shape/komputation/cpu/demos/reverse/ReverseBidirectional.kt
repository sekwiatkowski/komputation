package shape.komputation.cpu.demos.reverse

import shape.komputation.cpu.Network
import shape.komputation.cpu.layers.forward.units.simpleRecurrentUnit
import shape.komputation.cpu.printLoss
import shape.komputation.demos.reverse.ReverseData
import shape.komputation.initialization.gaussianInitialization
import shape.komputation.initialization.identityInitialization
import shape.komputation.initialization.zeroInitialization
import shape.komputation.layers.entry.inputLayer
import shape.komputation.layers.forward.activation.ActivationFunction
import shape.komputation.layers.forward.concatenation
import shape.komputation.layers.forward.decoder.singleInputDecoder
import shape.komputation.layers.forward.encoder.singleOutputEncoder
import shape.komputation.loss.logisticLoss
import shape.komputation.optimization.stochasticGradientDescent
import java.util.*

fun main(args: Array<String>) {

    val random = Random(1)
    val numberExamples = 10_000
    val seriesLength = 6
    val numberCategories = 10
    val hiddenDimension = 30
    val numberIterations = 100
    val batchSize = 1

    val inputs = ReverseData.generateInputs(random, numberExamples, seriesLength, numberCategories)
    val targets = ReverseData.generateTargets(inputs, seriesLength, numberCategories)

    val identityInitialization = identityInitialization()
    val gaussianInitialization = gaussianInitialization(random, 0.0f, 0.001f)
    val zeroInitialization = zeroInitialization()

    val optimizationStrategy = stochasticGradientDescent(0.0005f)

    val forwardEncoderUnit = simpleRecurrentUnit(
        seriesLength,
        numberCategories,
        hiddenDimension,
        gaussianInitialization,
        identityInitialization,
        zeroInitialization,
        ActivationFunction.ReLU,
        optimizationStrategy
    )

    val backwardEncoderUnit = simpleRecurrentUnit(
        seriesLength,
        numberCategories,
        hiddenDimension,
        gaussianInitialization,
        identityInitialization,
        zeroInitialization,
        ActivationFunction.ReLU,
        optimizationStrategy
    )

    val decoderUnit = simpleRecurrentUnit(
        seriesLength,
        numberCategories,
        2 * hiddenDimension,
        identityInitialization,
        gaussianInitialization,
        zeroInitialization,
        ActivationFunction.ReLU,
        optimizationStrategy
    )

    val network = Network(
        inputLayer(numberCategories, seriesLength),
        concatenation(
            numberCategories,
            seriesLength,
            true,
            intArrayOf(hiddenDimension, hiddenDimension),
            1,
            arrayOf(
                singleOutputEncoder(forwardEncoderUnit, seriesLength, numberCategories, hiddenDimension, false),
                singleOutputEncoder(backwardEncoderUnit, seriesLength, numberCategories, hiddenDimension, true)
            )
        ),
        singleInputDecoder(seriesLength, 2 * hiddenDimension, numberCategories, decoderUnit, gaussianInitialization, null, ActivationFunction.Softmax, optimizationStrategy)
    )

    network.train(
        inputs,
        targets,
        logisticLoss(numberCategories, seriesLength),
        numberIterations,
        batchSize,
        printLoss
    )

}

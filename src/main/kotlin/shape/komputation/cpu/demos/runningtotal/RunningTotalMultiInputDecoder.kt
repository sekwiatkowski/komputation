package shape.komputation.cpu.demos.runningtotal

import shape.komputation.cpu.Network
import shape.komputation.cpu.layers.forward.units.simpleRecurrentUnit
import shape.komputation.cpu.printLoss
import shape.komputation.demos.runningtotal.RunningTotalData
import shape.komputation.initialization.gaussianInitialization
import shape.komputation.initialization.identityInitialization
import shape.komputation.initialization.zeroInitialization
import shape.komputation.layers.entry.inputLayer
import shape.komputation.layers.forward.activation.ActivationFunction
import shape.komputation.layers.forward.decoder.multiInputDecoder
import shape.komputation.layers.forward.encoder.multiOutputEncoder
import shape.komputation.loss.squaredLoss
import shape.komputation.matrix.IntMath
import shape.komputation.optimization.stochasticGradientDescent
import java.util.*

fun main(args: Array<String>) {

    val exclusiveUpperLimit = 10
    val numberSteps = 4
    val numberExamples = IntMath.pow(exclusiveUpperLimit, numberSteps)
    val inputDimension = 1
    val hiddenDimension = 4
    val numberIterations = 30
    val batchSize = 4

    val random = Random(1)

    val identityInitialization = identityInitialization()
    val gaussianInitialization = gaussianInitialization(random, 0.0f, 0.001f)
    val zeroInitialization = zeroInitialization()

    val optimizationStrategy = stochasticGradientDescent(0.001f)

    val inputs = RunningTotalData.generateInputs(random, numberExamples, numberSteps, exclusiveUpperLimit)

    val targets = RunningTotalData.generateTargets(inputs)

    val encoderUnit = simpleRecurrentUnit(
        numberSteps,
        inputDimension,
        hiddenDimension,
        gaussianInitialization,
        identityInitialization,
        zeroInitialization,
        ActivationFunction.Identity,
        optimizationStrategy)

    val encoder = multiOutputEncoder(
        encoderUnit,
        numberSteps,
        inputDimension,
        hiddenDimension)

    val decoderUnit = simpleRecurrentUnit(
        numberSteps,
        hiddenDimension,
        hiddenDimension,
        identityInitialization,
        gaussianInitialization,
        zeroInitialization,
        ActivationFunction.Identity,
        optimizationStrategy
    )

    val decoder = multiInputDecoder(
        numberSteps,
        hiddenDimension,
        hiddenDimension,
        1,
        decoderUnit,
        gaussianInitialization,
        zeroInitialization,
        ActivationFunction.Identity,
        optimizationStrategy
    )

    Network(
            batchSize,
            inputLayer(numberSteps),
            encoder,
            decoder
        )
        .training(
            inputs,
            targets,
            numberIterations,
            squaredLoss(numberSteps),
            printLoss
        )
        .run()

}
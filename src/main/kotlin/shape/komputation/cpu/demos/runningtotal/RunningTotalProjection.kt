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
import shape.komputation.layers.forward.encoder.multiOutputEncoder
import shape.komputation.layers.forward.projection.projectionLayer
import shape.komputation.loss.squaredLoss
import shape.komputation.optimization.stochasticGradientDescent
import java.util.*

fun main(args: Array<String>) {

    val numberSteps = 4
    val exclusiveUpperLimit = 10
    val hiddenDimension = 4
    val outputDimension = 1
    val numberExamples = Math.pow(exclusiveUpperLimit.toDouble(), numberSteps.toDouble()).toInt()
    val numberIterations = 30
    val batchSize = 4

    val random = Random(1)

    val previousStateWeightInitializationStrategy = identityInitialization()
    val inputWeightInitializationStrategy = gaussianInitialization(random, 0.0, 0.001)
    val biasInitializationStrategy = zeroInitialization()

    val optimizationStrategy = stochasticGradientDescent(0.001)

    val inputs = RunningTotalData.generateInputs(random, numberExamples, numberSteps, exclusiveUpperLimit)

    val targets = RunningTotalData.generateTargets(inputs, numberSteps)

    val encoderUnit = simpleRecurrentUnit(
        numberSteps,
        hiddenDimension,
        1,
        inputWeightInitializationStrategy,
        previousStateWeightInitializationStrategy,
        biasInitializationStrategy,
        ActivationFunction.Identity,
        optimizationStrategy
    )

    val encoder = multiOutputEncoder(encoderUnit, numberSteps, 1, hiddenDimension)

    val network = Network(
        inputLayer(numberSteps),
        encoder,
        projectionLayer(hiddenDimension, outputDimension, inputWeightInitializationStrategy, inputWeightInitializationStrategy, optimizationStrategy)
    )

    network.train(
        inputs,
        targets,
        squaredLoss(1),
        numberIterations,
        batchSize,
        printLoss
    )

}

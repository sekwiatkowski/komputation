package shape.komputation.demos.runningtotal

import shape.komputation.functions.activation.ActivationFunction
import shape.komputation.initialization.gaussianInitialization
import shape.komputation.initialization.identityInitialization
import shape.komputation.initialization.zeroInitialization
import shape.komputation.layers.entry.inputLayer
import shape.komputation.layers.forward.encoder.multiOutputEncoder
import shape.komputation.layers.forward.projection.projectionLayer
import shape.komputation.layers.forward.units.simpleRecurrentUnit
import shape.komputation.loss.squaredLoss
import shape.komputation.networks.Network
import shape.komputation.networks.printLoss
import shape.komputation.optimization.stochasticGradientDescent
import java.util.*

fun main(args: Array<String>) {

    val numberSteps = 4
    val exclusiveUpperLimit = 10
    val hiddenDimension = 4
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
        inputLayer(),
        encoder,
        projectionLayer(hiddenDimension, 1, inputWeightInitializationStrategy, inputWeightInitializationStrategy, optimizationStrategy)
    )

    network.train(
        inputs,
        targets,
        squaredLoss(),
        numberIterations,
        batchSize,
        printLoss
    )

}

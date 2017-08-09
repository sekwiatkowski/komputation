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
import shape.komputation.matrix.IntMath
import shape.komputation.optimization.stochasticGradientDescent
import java.util.*

fun main(args: Array<String>) {

    val numberSteps = 4
    val exclusiveUpperLimit = 10
    val inputDimension = 1
    val hiddenDimension = 4
    val outputDimension = 1
    val numberExamples = IntMath.pow(exclusiveUpperLimit, numberSteps)
    val numberIterations = 30
    val batchSize = 4

    val random = Random(1)

    val identityInitialization = identityInitialization()
    val guassianInitialization = gaussianInitialization(random, 0.0f, 0.001f)
    val zeroInitialization = zeroInitialization()

    val optimizationStrategy = stochasticGradientDescent(0.001f)

    val inputs = RunningTotalData.generateInputs(random, numberExamples, numberSteps, exclusiveUpperLimit)

    val targets = RunningTotalData.generateTargets(inputs)

    val encoderUnit = simpleRecurrentUnit(
        numberSteps,
        inputDimension,
        hiddenDimension,
        guassianInitialization,
        identityInitialization,
        zeroInitialization,
        ActivationFunction.Identity,
        optimizationStrategy
    )

    val network = Network(
        inputLayer(inputDimension, numberSteps),
        multiOutputEncoder(encoderUnit, numberSteps, inputDimension, hiddenDimension),
        projectionLayer(hiddenDimension, numberSteps, false, outputDimension, guassianInitialization, guassianInitialization, optimizationStrategy)
    )

    network.train(
        inputs,
        targets,
        squaredLoss(numberSteps),
        numberIterations,
        batchSize,
        printLoss
    )

}

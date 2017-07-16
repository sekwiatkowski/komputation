package shape.komputation.cpu.demos.and

import shape.komputation.cpu.Network
import shape.komputation.cpu.printLoss
import shape.komputation.demos.and.BinaryAndData
import shape.komputation.initialization.heInitialization
import shape.komputation.layers.entry.inputLayer
import shape.komputation.layers.forward.activation.sigmoidLayer
import shape.komputation.layers.forward.projection.projectionLayer
import shape.komputation.loss.squaredLoss
import shape.komputation.optimization.stochasticGradientDescent
import java.util.*

fun main(args: Array<String>) {

    val random = Random(1)

    val inputDimension = 2
    val outputDimension = 1

    val initialize = heInitialization(random)
    val optimizer = stochasticGradientDescent(0.03)

    val projectionLayer = projectionLayer(inputDimension, outputDimension, initialize, initialize, optimizer)

    val network = Network(
        inputLayer(inputDimension),
        projectionLayer,
        sigmoidLayer(outputDimension)
    )

    network.train(BinaryAndData.inputs, BinaryAndData.targets, squaredLoss(outputDimension), 10_000, 1, printLoss)

}
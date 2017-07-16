package shape.komputation.cpu.demos.and

import shape.komputation.cpu.Network
import shape.komputation.cpu.printLoss
import shape.komputation.demos.and.OneHotAndData
import shape.komputation.initialization.heInitialization
import shape.komputation.layers.entry.inputLayer
import shape.komputation.layers.forward.activation.softmaxLayer
import shape.komputation.layers.forward.projection.projectionLayer
import shape.komputation.loss.logisticLoss
import shape.komputation.optimization.stochasticGradientDescent
import java.util.*

fun main(args: Array<String>) {

    val inputDimension = 2
    val outputDimension = 2

    val random = Random(1)
    val initialize = heInitialization(random)

    val optimizer = stochasticGradientDescent(0.03)

    val projectionLayer = projectionLayer(inputDimension, outputDimension, initialize, initialize, optimizer)
    val softmaxLayer = softmaxLayer()

    val network = Network(
        inputLayer(inputDimension),
        projectionLayer,
        softmaxLayer
    )

    network.train(OneHotAndData.input, OneHotAndData.targets, logisticLoss(), 10_000, 1, printLoss)

}
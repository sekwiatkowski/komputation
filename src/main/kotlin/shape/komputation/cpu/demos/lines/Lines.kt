package shape.komputation.cpu.demos.lines

import shape.komputation.cpu.Network
import shape.komputation.cpu.printLoss
import shape.komputation.demos.lines.LinesData
import shape.komputation.initialization.uniformInitialization
import shape.komputation.layers.entry.inputLayer
import shape.komputation.layers.forward.activation.ActivationFunction
import shape.komputation.layers.forward.activation.reluLayer
import shape.komputation.layers.forward.convolution.convolutionalLayer
import shape.komputation.layers.forward.convolution.maxPoolingLayer
import shape.komputation.layers.forward.denseLayer
import shape.komputation.loss.logisticLoss
import shape.komputation.optimization.stochasticGradientDescent
import java.util.*

fun main(args: Array<String>) {

    val numberRows = 3
    val numberColumns = 3

    val filterWidth = 3
    val filterHeight = 1
    val numberFilters = 6

    val outputDimension = 2

    val random = Random(1)
    val initialize = uniformInitialization(random, -0.05f, 0.05f)

    val optimization = stochasticGradientDescent(0.01f)

    val maximumBatchSize = 1

    val network = Network(
        inputLayer(numberRows),
        convolutionalLayer(numberRows, numberColumns, numberFilters, filterWidth, filterHeight, initialize, initialize, optimization),
        maxPoolingLayer(numberFilters, 3),
        reluLayer(numberFilters),
        denseLayer(numberFilters, outputDimension, initialize, initialize, ActivationFunction.Softmax, optimization)
    )

    network.train(LinesData.inputs, LinesData.targets, logisticLoss(outputDimension), 30_000, maximumBatchSize, printLoss)

}
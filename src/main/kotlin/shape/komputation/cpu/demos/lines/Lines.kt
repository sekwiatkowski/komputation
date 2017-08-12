package shape.komputation.cpu.demos.lines

import shape.komputation.cpu.Network
import shape.komputation.cpu.printLoss
import shape.komputation.demos.lines.LinesData
import shape.komputation.initialization.uniformInitialization
import shape.komputation.layers.entry.inputLayer
import shape.komputation.layers.forward.activation.ActivationFunction
import shape.komputation.layers.forward.activation.reluLayer
import shape.komputation.layers.forward.convolution.convolutionalLayer
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

    Network(
            maximumBatchSize,
            inputLayer(numberRows, numberColumns),
            convolutionalLayer(numberRows, numberColumns, true, numberFilters, filterWidth, filterHeight, initialize, initialize, optimization),
            reluLayer(numberFilters),
            denseLayer(numberFilters, outputDimension, initialize, initialize, ActivationFunction.Softmax, optimization)
        )
        .training(
            LinesData.inputs,
            LinesData.targets,
            30_000,
            logisticLoss(outputDimension),
            printLoss)
        .run()

}
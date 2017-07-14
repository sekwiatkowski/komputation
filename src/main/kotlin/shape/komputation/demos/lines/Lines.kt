package shape.komputation.demos.lines

import shape.komputation.cpu.loss.logisticLoss
import shape.komputation.initialization.uniformInitialization
import shape.komputation.layers.entry.inputLayer
import shape.komputation.layers.forward.activation.reluLayer
import shape.komputation.layers.forward.activation.softmaxLayer
import shape.komputation.layers.forward.convolution.convolutionalLayer
import shape.komputation.layers.forward.convolution.maxPoolingLayer
import shape.komputation.layers.forward.projection.projectionLayer
import shape.komputation.matrix.Matrix
import shape.komputation.matrix.doubleColumnVector
import shape.komputation.matrix.doubleMatrixFromRows
import shape.komputation.networks.Network
import shape.komputation.networks.printLoss
import shape.komputation.optimization.stochasticGradientDescent
import java.util.*

object LinesData {

    val inputs = arrayOf<Matrix>(

        doubleMatrixFromRows(
            doubleArrayOf(1.0, 1.0, 1.0),
            doubleArrayOf(0.0, 0.0, 0.0),
            doubleArrayOf(0.0, 0.0, 0.0)
        ),
        doubleMatrixFromRows(
            doubleArrayOf(0.0, 0.0, 0.0),
            doubleArrayOf(1.0, 1.0, 1.0),
            doubleArrayOf(0.0, 0.0, 0.0)
        ),
        doubleMatrixFromRows(
            doubleArrayOf(0.0, 0.0, 0.0),
            doubleArrayOf(0.0, 0.0, 0.0),
            doubleArrayOf(1.0, 1.0, 1.0)
        ),
        doubleMatrixFromRows(
            doubleArrayOf(1.0, 0.0, 0.0),
            doubleArrayOf(1.0, 0.0, 0.0),
            doubleArrayOf(1.0, 0.0, 0.0)
        ),
        doubleMatrixFromRows(
            doubleArrayOf(0.0, 1.0, 0.0),
            doubleArrayOf(0.0, 1.0, 0.0),
            doubleArrayOf(0.0, 1.0, 0.0)
        ),
        doubleMatrixFromRows(
            doubleArrayOf(0.0, 0.0, 1.0),
            doubleArrayOf(0.0, 0.0, 1.0),
            doubleArrayOf(0.0, 0.0, 1.0)
        )
    )

    val targets = arrayOf(
        doubleColumnVector(0.0, 1.0),
        doubleColumnVector(0.0, 1.0),
        doubleColumnVector(0.0, 1.0),
        doubleColumnVector(1.0, 0.0),
        doubleColumnVector(1.0, 0.0),
        doubleColumnVector(1.0, 0.0)
    )

}

fun main(args: Array<String>) {

    val random = Random(1)
    val initialize = uniformInitialization(random, -0.05, 0.05)

    val optimization = stochasticGradientDescent(0.01)

    val filterWidth = 3
    val filterHeight = 1
    val numberFilters = 6

    val network = Network(
        inputLayer(),
        convolutionalLayer(numberFilters, filterWidth, filterHeight, initialize, optimization),
        maxPoolingLayer(),
        reluLayer(),
        projectionLayer(numberFilters, 2, initialize, initialize, optimization),
        softmaxLayer()
    )

    network.train(LinesData.inputs, LinesData.targets, logisticLoss(), 30_000, 1, printLoss)
}
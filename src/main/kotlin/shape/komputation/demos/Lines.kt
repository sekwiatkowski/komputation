package shape.komputation.demos

import shape.komputation.*
import shape.komputation.initialization.createUniformInitializer
import shape.komputation.layers.continuation.activation.ReluLayer
import shape.komputation.layers.continuation.activation.SoftmaxLayer
import shape.komputation.layers.continuation.*
import shape.komputation.layers.continuation.convolution.MaxPoolingLayer
import shape.komputation.layers.continuation.convolution.createConvolutionalLayer
import shape.komputation.layers.entry.InputLayer
import shape.komputation.loss.LogisticLoss
import shape.komputation.matrix.Matrix
import shape.komputation.matrix.createRealMatrix
import shape.komputation.optimization.stochasticGradientDescent
import java.util.*

fun main(args: Array<String>) {

    val input = arrayOf<Matrix>(

        createRealMatrix(
            doubleArrayOf(1.0, 1.0, 1.0),
            doubleArrayOf(0.0, 0.0, 0.0),
            doubleArrayOf(0.0, 0.0, 0.0)
        ),
        createRealMatrix(
            doubleArrayOf(0.0, 0.0, 0.0),
            doubleArrayOf(1.0, 1.0, 1.0),
            doubleArrayOf(0.0, 0.0, 0.0)
        ),
        createRealMatrix(
            doubleArrayOf(0.0, 0.0, 0.0),
            doubleArrayOf(0.0, 0.0, 0.0),
            doubleArrayOf(1.0, 1.0, 1.0)
        ),
        createRealMatrix(
            doubleArrayOf(1.0, 0.0, 0.0),
            doubleArrayOf(1.0, 0.0, 0.0),
            doubleArrayOf(1.0, 0.0, 0.0)
        ),
        createRealMatrix(
            doubleArrayOf(0.0, 1.0, 0.0),
            doubleArrayOf(0.0, 1.0, 0.0),
            doubleArrayOf(0.0, 1.0, 0.0)
        ),
        createRealMatrix(
            doubleArrayOf(0.0, 0.0, 1.0),
            doubleArrayOf(0.0, 0.0, 1.0),
            doubleArrayOf(0.0, 0.0, 1.0)
        )
    )

    val targets = arrayOf(
        createRealMatrix(
            doubleArrayOf(0.0),
            doubleArrayOf(1.0)
        ),
        createRealMatrix(
            doubleArrayOf(0.0),
            doubleArrayOf(1.0)
        ),
        createRealMatrix(
            doubleArrayOf(0.0),
            doubleArrayOf(1.0)
        ),
        createRealMatrix(
            doubleArrayOf(1.0),
            doubleArrayOf(0.0)
        ),
        createRealMatrix(
            doubleArrayOf(1.0),
            doubleArrayOf(0.0)
        ),
        createRealMatrix(
            doubleArrayOf(1.0),
            doubleArrayOf(0.0)
        )
    )

    val random = Random(1)
    val initialize = createUniformInitializer(random, -0.05, 0.05)

    val optimization = stochasticGradientDescent(0.01)

    val filterWidth = 3
    val filterHeight = 1
    val numberFilters = 6

    val network = Network(
        InputLayer(),
        createConvolutionalLayer(numberFilters, filterWidth, filterHeight, initialize, optimization),
        MaxPoolingLayer(),
        ReluLayer(),
        createProjectionLayer(numberFilters, 2, initialize, optimization),
        SoftmaxLayer()
    )

    train(network, input, targets, LogisticLoss(), 30_000, printLoss)
}
package shape.konvolution.demos

import shape.konvolution.*
import shape.konvolution.layers.continuation.*
import shape.konvolution.layers.entry.InputLayer
import shape.konvolution.loss.LogisticLoss
import shape.konvolution.matrix.Matrix
import shape.konvolution.matrix.RealMatrix
import shape.konvolution.matrix.createRealMatrix
import shape.konvolution.optimization.stochasticGradientDescent
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

    val targets = arrayOf<RealMatrix>(
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

    val updateRule = stochasticGradientDescent(0.01)

    val filterWidth = 3
    val filterHeight = 1
    val numberFilters = 6

    val network = Network(
        InputLayer(),
        ExpansionLayer(filterWidth, filterHeight),
        createProjectionLayer(filterWidth * filterHeight, numberFilters, initialize, updateRule),
        ReluLayer(),
        MaxPoolingLayer(),
        createProjectionLayer(numberFilters, 2, initialize, updateRule),
        SoftmaxLayer()
    )

    train(network, input, targets, LogisticLoss(), 30_000)

}
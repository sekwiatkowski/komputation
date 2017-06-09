package shape.konvolution.demos

import shape.konvolution.*
import shape.konvolution.layers.continuation.*
import shape.konvolution.layers.entry.InputLayer
import shape.konvolution.loss.LogisticLoss
import shape.konvolution.optimization.StochasticGradientDescent
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

    val optimizer = StochasticGradientDescent(0.01)

    val network = Network(
        InputLayer(),
        createConvolutionLayer(6, 3, 1, initialize, optimizer, optimizer),
        MaxPoolingLayer(),
        ReluLayer(),
        createProjectionLayer(6, 2, initialize, optimizer, optimizer),
        SoftmaxLayer()
    )

    train(network, input, targets, LogisticLoss(), 30_000)

}
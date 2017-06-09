package shape.konvolution.demos

import no.uib.cipr.matrix.Matrix
import shape.konvolution.Network
import shape.konvolution.createDenseMatrix
import shape.konvolution.createUniformInitializer
import shape.konvolution.layers.*
import shape.konvolution.loss.LogisticLoss
import shape.konvolution.loss.SquaredLoss
import shape.konvolution.optimization.StochasticGradientDescent
import shape.konvolution.train
import java.util.*

fun main(args: Array<String>) {

    val input = arrayOf<Matrix>(

        createDenseMatrix(
            doubleArrayOf(1.0, 1.0, 1.0),
            doubleArrayOf(0.0, 0.0, 0.0),
            doubleArrayOf(0.0, 0.0, 0.0)
        ),
        createDenseMatrix(
            doubleArrayOf(0.0, 0.0, 0.0),
            doubleArrayOf(1.0, 1.0, 1.0),
            doubleArrayOf(0.0, 0.0, 0.0)
        ),
        createDenseMatrix(
            doubleArrayOf(0.0, 0.0, 0.0),
            doubleArrayOf(0.0, 0.0, 0.0),
            doubleArrayOf(1.0, 1.0, 1.0)
        ),
        createDenseMatrix(
            doubleArrayOf(1.0, 0.0, 0.0),
            doubleArrayOf(1.0, 0.0, 0.0),
            doubleArrayOf(1.0, 0.0, 0.0)
        ),
        createDenseMatrix(
            doubleArrayOf(0.0, 1.0, 0.0),
            doubleArrayOf(0.0, 1.0, 0.0),
            doubleArrayOf(0.0, 1.0, 0.0)
        ),
        createDenseMatrix(
            doubleArrayOf(0.0, 0.0, 1.0),
            doubleArrayOf(0.0, 0.0, 1.0),
            doubleArrayOf(0.0, 0.0, 1.0)
        )
    )

    val targets = arrayOf<Matrix>(
        createDenseMatrix(
            doubleArrayOf(0.0),
            doubleArrayOf(1.0)
        ),
        createDenseMatrix(
            doubleArrayOf(0.0),
            doubleArrayOf(1.0)
        ),
        createDenseMatrix(
            doubleArrayOf(0.0),
            doubleArrayOf(1.0)
        ),
        createDenseMatrix(
            doubleArrayOf(1.0),
            doubleArrayOf(0.0)
        ),
        createDenseMatrix(
            doubleArrayOf(1.0),
            doubleArrayOf(0.0)
        ),
        createDenseMatrix(
            doubleArrayOf(1.0),
            doubleArrayOf(0.0)
        )
    )

    val random = Random(1)
    val initialize = createUniformInitializer(random, -0.05, 0.05)

    val optimizer = StochasticGradientDescent(0.01)

    val network = Network(
        arrayOf(
            createConvolutionLayer(6, 3, 1, initialize, optimizer, optimizer),
            MaxPoolingLayer(),
            ReluLayer(),
            createProjectionLayer(6, 2, initialize, optimizer, optimizer),
            SoftmaxLayer()
        )
    )

    train(network, input, targets, LogisticLoss(), 30_000)

}
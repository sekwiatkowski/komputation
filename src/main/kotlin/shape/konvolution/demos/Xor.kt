package shape.konvolution.demos

import no.uib.cipr.matrix.Matrix
import shape.konvolution.Network
import shape.konvolution.createDenseMatrix
import shape.konvolution.createUniformInitializer
import shape.konvolution.layers.SigmoidLayer
import shape.konvolution.layers.createProjectionLayer
import shape.konvolution.loss.SquaredLoss
import shape.konvolution.optimization.StochasticGradientDescent
import shape.konvolution.train
import java.util.*

fun main(args: Array<String>) {

    val input = arrayOf<Matrix>(

        createDenseMatrix(doubleArrayOf(0.0), doubleArrayOf(0.0)),
        createDenseMatrix(doubleArrayOf(1.0), doubleArrayOf(0.0)),
        createDenseMatrix(doubleArrayOf(0.0), doubleArrayOf(1.0)),
        createDenseMatrix(doubleArrayOf(1.0), doubleArrayOf(1.0))

    )

    val targets = arrayOf<Matrix>(
        createDenseMatrix(doubleArrayOf(0.0)),
        createDenseMatrix(doubleArrayOf(1.0)),
        createDenseMatrix(doubleArrayOf(1.0)),
        createDenseMatrix(doubleArrayOf(0.0))
    )

    val random = Random(1)
    val initialize = createUniformInitializer(random, -0.5, 0.5)

    val hiddenPreactivationLayer = createProjectionLayer(2, 2, initialize, StochasticGradientDescent(0.1), StochasticGradientDescent(0.1))
    val hiddenLayer = SigmoidLayer()

    val outputPreactivationLayer = createProjectionLayer(2, 1, initialize, StochasticGradientDescent(0.1), StochasticGradientDescent(0.1))
    val outputLayer = SigmoidLayer()

    val network = Network(
        arrayOf(
            hiddenPreactivationLayer,
            hiddenLayer,
            outputPreactivationLayer,
            outputLayer
        )
    )

    train(network, input, targets, SquaredLoss(), 30_000)

}
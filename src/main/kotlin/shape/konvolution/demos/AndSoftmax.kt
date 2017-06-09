package shape.konvolution.demos

import no.uib.cipr.matrix.Matrix
import shape.konvolution.Network
import shape.konvolution.createDenseMatrix
import shape.konvolution.createGaussianInitializer
import shape.konvolution.layers.SoftmaxLayer
import shape.konvolution.layers.createProjectionLayer
import shape.konvolution.loss.LogisticLoss
import shape.konvolution.optimization.StochasticGradientDescent
import shape.konvolution.train
import java.util.*

fun main(args: Array<String>) {

    val input = arrayOf<Matrix>(

        createDenseMatrix(
            doubleArrayOf(0.0),
            doubleArrayOf(0.0)
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
            doubleArrayOf(1.0)
        )

    )

    val targets = arrayOf<Matrix>(
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
        ),
        createDenseMatrix(
            doubleArrayOf(0.0),
            doubleArrayOf(1.0)
        )

    )

    val random = Random(1)
    val initialize = createGaussianInitializer(random)

    val projectionLayer = createProjectionLayer(2, 2, initialize, StochasticGradientDescent(0.03), StochasticGradientDescent(0.03))
    val softmaxLayer = SoftmaxLayer()

    val network = Network(
        arrayOf(
            projectionLayer,
            softmaxLayer
        )
    )

    train(network, input, targets, LogisticLoss(), 10_000)


}
package shape.komputation.demos

import shape.komputation.*
import shape.komputation.layers.entry.InputLayer
import shape.komputation.layers.continuation.SigmoidLayer
import shape.komputation.layers.continuation.createProjectionLayer
import shape.komputation.loss.SquaredLoss
import shape.komputation.matrix.Matrix
import shape.komputation.matrix.RealMatrix
import shape.komputation.matrix.createRealMatrix
import shape.komputation.optimization.stochasticGradientDescent
import java.util.*

fun main(args: Array<String>) {

    val input = arrayOf<Matrix>(

        createRealMatrix(
            doubleArrayOf(0.0),
            doubleArrayOf(0.0)
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
            doubleArrayOf(1.0)
        )
    )

    val targets = arrayOf<RealMatrix>(
        createRealMatrix(
            doubleArrayOf(0.0)
        ),
        createRealMatrix(
            doubleArrayOf(0.0)
        ),
        createRealMatrix(
            doubleArrayOf(0.0)
        ),
        createRealMatrix(
            doubleArrayOf(1.0)
        )
    )

    val random = Random(1)
    val initialize = createGaussianInitializer(random)

    val optimizer = stochasticGradientDescent(0.03)

    val projectionLayer = createProjectionLayer(2, 1, initialize, optimizer)
    val sigmoidLayer = SigmoidLayer()

    val network = Network(
        InputLayer(),
        projectionLayer,
        sigmoidLayer
    )

    train(network, input, targets, SquaredLoss(), 10_000, printLoss)

}
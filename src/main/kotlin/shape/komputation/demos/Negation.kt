package shape.komputation.demos

import shape.komputation.*
import shape.komputation.initialization.createUniformInitializer
import shape.komputation.layers.entry.InputLayer
import shape.komputation.layers.continuation.activation.SigmoidLayer
import shape.komputation.layers.continuation.createProjectionLayer
import shape.komputation.loss.SquaredLoss
import shape.komputation.matrix.Matrix
import shape.komputation.matrix.createRealMatrix
import shape.komputation.optimization.stochasticGradientDescent
import java.util.*

fun main(args: Array<String>) {

    val input = arrayOf<Matrix>(

        createRealMatrix(
            doubleArrayOf(0.0)
        ),
        createRealMatrix(
            doubleArrayOf(1.0)
        )

    )

    val targets = arrayOf(
        createRealMatrix(
            doubleArrayOf(1.0)
        ),
        createRealMatrix(
            doubleArrayOf(0.0)
        )
    )

    val random = Random(1)
    val initialize = createUniformInitializer(random, -0.5, 0.5)

    val updateRule = stochasticGradientDescent(0.01)

    val projectionLayer = createProjectionLayer(1, 1, initialize, updateRule)
    val sigmoidLayer = SigmoidLayer()

    val network = Network(
        InputLayer(),
        projectionLayer,
        sigmoidLayer
    )

    train(network, input, targets, SquaredLoss(), 10_000, printLoss)

}
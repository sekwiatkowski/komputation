package shape.konvolution.demos

import shape.konvolution.*
import shape.konvolution.layers.entry.InputLayer
import shape.konvolution.layers.continuation.SigmoidLayer
import shape.konvolution.layers.continuation.createProjectionLayer
import shape.konvolution.loss.SquaredLoss
import shape.konvolution.optimization.StochasticGradientDescent
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

    val optimizer = StochasticGradientDescent(0.03)
    val projectionLayer = createProjectionLayer(2, 1, initialize, optimizer, optimizer)
    val sigmoidLayer = SigmoidLayer()

    val network = Network(
        InputLayer(),
        projectionLayer,
        sigmoidLayer
    )

    train(network, input, targets, SquaredLoss(), 10_000)


}
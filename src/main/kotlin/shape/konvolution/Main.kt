package shape.konvolution

import shape.konvolution.layers.*
import shape.konvolution.optimization.StochasticGradientDescent
import java.util.*




fun main(args: Array<String>) {

    val embeddingDimension = 300
    val numberClasses = 6
    val numberFilters = 100

    val random = Random(1)

    val inputMatrix = createDenseMatrix(doubleArrayOf(0.0, 0.0))

    val initialize = createUniformInitializer(random, -0.5, 0.5)

    val createBranch = { filterHeight : Int ->

        val convLayer = createConvolutionLayer(numberFilters, embeddingDimension, filterHeight, initialize, StochasticGradientDescent(0.01), StochasticGradientDescent(0.01))

        val maxPoolLayer = MaxPoolingLayer()

        arrayOf(convLayer, maxPoolLayer)

    }

    val convolutionBranch = Branch(
        3 * 100,
        createBranch(3),
        createBranch(4),
        createBranch(5)
    )

    val reluLayer = ReluLayer()

    val projectionLayer = createProjectionLayer(3 * 100, numberClasses, initialize, StochasticGradientDescent(0.01), StochasticGradientDescent(0.01))

    val outputLayer = SoftmaxLayer()

    val cnn = Network(arrayOf(convolutionBranch, reluLayer, projectionLayer, outputLayer))


}


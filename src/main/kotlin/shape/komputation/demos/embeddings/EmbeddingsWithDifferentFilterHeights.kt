package shape.komputation.demos.embeddings

import shape.komputation.initialization.createUniformInitializer
import shape.komputation.initialization.initializeColumnVector
import shape.komputation.layers.entry.createLookupLayer
import shape.komputation.layers.forward.activation.ReluLayer
import shape.komputation.layers.forward.activation.SoftmaxLayer
import shape.komputation.layers.forward.convolution.MaxPoolingLayer
import shape.komputation.layers.forward.convolution.createConvolutionalLayer
import shape.komputation.layers.forward.createConcatenation
import shape.komputation.layers.forward.projection.createProjectionLayer
import shape.komputation.loss.SquaredLoss
import shape.komputation.networks.Network
import shape.komputation.networks.printLoss
import shape.komputation.optimization.momentum
import java.util.*

fun main(args: Array<String>) {

    val random = Random(1)

    val maximumBatchSize = 1
    val numberEmbeddings = 40
    val embeddingDimension = 2

    val initializationStrategy = createUniformInitializer(random, -0.05, 0.05)

    val initializeEmbedding = { initializeColumnVector(initializationStrategy, embeddingDimension) }
    val embeddings = Array(numberEmbeddings) { initializeEmbedding() }

    val optimizationStrategy = momentum(0.01, 0.9)

    val numberFilters = 2

    val filterWidth = embeddingDimension
    val filterHeights = arrayOf(1, 2)

    val createConvolutionSubnetwork = { filterHeight : Int ->

        arrayOf(
            createConvolutionalLayer(numberFilters, filterWidth, filterHeight, initializationStrategy, optimizationStrategy),
            ReluLayer(),
            MaxPoolingLayer()

        )

    }

    val numberClasses = EmbeddingData.numberClasses
    val input = EmbeddingData.inputs
    val targets = EmbeddingData.targets

    val network = Network(
        createLookupLayer(embeddings, embeddingDimension, maximumBatchSize, optimizationStrategy),
        createConcatenation(
            *filterHeights.map { filterHeight -> createConvolutionSubnetwork(filterHeight) }.toTypedArray()
        ),
        createProjectionLayer(numberFilters * filterHeights.size, numberClasses, initializationStrategy, initializationStrategy, optimizationStrategy),
        SoftmaxLayer()
    )

    network.train(input, targets, SquaredLoss(), 10_000, maximumBatchSize, printLoss)

}
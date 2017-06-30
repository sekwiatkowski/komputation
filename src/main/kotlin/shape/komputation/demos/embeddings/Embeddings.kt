package shape.komputation.demos.embeddings

import shape.komputation.initialization.createUniformInitializer
import shape.komputation.initialization.initializeColumnVector
import shape.komputation.layers.entry.createLookupLayer
import shape.komputation.layers.feedforward.activation.ReluLayer
import shape.komputation.layers.feedforward.activation.SoftmaxLayer
import shape.komputation.layers.feedforward.convolution.MaxPoolingLayer
import shape.komputation.layers.feedforward.convolution.createConvolutionalLayer
import shape.komputation.layers.feedforward.projection.createProjectionLayer
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
    val filterHeight = 2

    val inputs = EmbeddingData.inputs
    val targets = EmbeddingData.targets
    val numberClasses = EmbeddingData.numberClasses

    val network = Network(
        createLookupLayer(embeddings, embeddingDimension, maximumBatchSize, optimizationStrategy),
        createConvolutionalLayer(numberFilters, filterWidth, filterHeight, initializationStrategy, optimizationStrategy),
        MaxPoolingLayer(),
        ReluLayer(),
        createProjectionLayer(numberFilters, numberClasses, true, initializationStrategy, optimizationStrategy),
        SoftmaxLayer()
    )

    network.train(inputs, targets, SquaredLoss(), 5_000, maximumBatchSize, printLoss)

}
package shape.komputation.demos.embeddings

import shape.komputation.initialization.uniformInitialization
import shape.komputation.initialization.initializeColumnVector
import shape.komputation.layers.entry.lookupLayer
import shape.komputation.layers.forward.activation.reluLayer
import shape.komputation.layers.forward.activation.softmaxLayer
import shape.komputation.layers.forward.convolution.convolutionalLayer
import shape.komputation.layers.forward.convolution.maxPoolingLayer
import shape.komputation.layers.forward.projection.projectionLayer
import shape.komputation.loss.squaredLoss
import shape.komputation.networks.Network
import shape.komputation.networks.printLoss
import shape.komputation.optimization.momentum
import java.util.*

fun main(args: Array<String>) {

    val random = Random(1)

    val maximumBatchSize = 1
    val numberEmbeddings = 40
    val embeddingDimension = 2

    val initializationStrategy = uniformInitialization(random, -0.05, 0.05)

    val initializeEmbedding = { initializeColumnVector(initializationStrategy, embeddingDimension) }
    val embeddings = Array(numberEmbeddings) { initializeEmbedding() }

    val optimizationStrategy = momentum(0.01, 0.9)

    val numberFilters = 2

    val filterWidth = 2
    val filterHeight = embeddingDimension

    val inputs = EmbeddingData.inputs
    val targets = EmbeddingData.targets
    val numberClasses = EmbeddingData.numberClasses

    val network = Network(
        lookupLayer(embeddings, embeddingDimension, maximumBatchSize, 2, optimizationStrategy),
        convolutionalLayer(numberFilters, filterWidth, filterHeight, initializationStrategy, optimizationStrategy),
        maxPoolingLayer(),
        reluLayer(),
        projectionLayer(numberFilters, numberClasses, initializationStrategy, initializationStrategy, optimizationStrategy),
        softmaxLayer()
    )

    network.train(inputs, targets, squaredLoss(), 5_000, maximumBatchSize, printLoss)

}
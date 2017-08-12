package shape.komputation.cpu.demos.embeddings

import shape.komputation.cpu.Network
import shape.komputation.cpu.printLoss
import shape.komputation.demos.embeddings.EmbeddingData
import shape.komputation.initialization.initializeColumnVector
import shape.komputation.initialization.uniformInitialization
import shape.komputation.layers.entry.lookupLayer
import shape.komputation.layers.forward.activation.ActivationFunction
import shape.komputation.layers.forward.activation.reluLayer
import shape.komputation.layers.forward.convolution.convolutionalLayer
import shape.komputation.layers.forward.denseLayer
import shape.komputation.loss.squaredLoss
import shape.komputation.optimization.historical.momentum
import java.util.*

fun main(args: Array<String>) {

    val random = Random(1)

    val maximumBatchSize = 1
    val numberEmbeddings = 40
    val embeddingDimension = 2

    val initializationStrategy = uniformInitialization(random, -0.05f, 0.05f)

    val initializeEmbedding = { initializeColumnVector(initializationStrategy, embeddingDimension) }
    val embeddings = Array(numberEmbeddings) { initializeEmbedding() }

    val optimizationStrategy = momentum(0.01f, 0.9f)

    val numberFilters = 2

    val filterWidth = 2
    val filterHeight = embeddingDimension

    val inputs = EmbeddingData.inputs
    val targets = EmbeddingData.targets
    val numberClasses = EmbeddingData.numberClasses

    val hasFixedLength = true

    Network(
            maximumBatchSize,
            lookupLayer(embeddings, 2, hasFixedLength, embeddingDimension, maximumBatchSize, optimizationStrategy),
            convolutionalLayer(embeddingDimension, 2, hasFixedLength, numberFilters, filterWidth, filterHeight, initializationStrategy, initializationStrategy, optimizationStrategy),
            reluLayer(numberFilters),
            denseLayer(numberFilters, numberClasses, initializationStrategy, initializationStrategy, ActivationFunction.Softmax, optimizationStrategy)
        )
        .training(
            inputs,
            targets,
            5_000,
            squaredLoss(numberClasses),
            printLoss)
        .run()

}
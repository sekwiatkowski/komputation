package com.komputation.cuda.demos.embeddings

import com.komputation.loss.printLoss
import com.komputation.cuda.network.CudaNetwork
import com.komputation.demos.embeddings.EmbeddingData
import com.komputation.initialization.initializeColumnVector
import com.komputation.initialization.uniformInitialization
import com.komputation.instructions.entry.lookup
import com.komputation.instructions.continuation.activation.Activation
import com.komputation.instructions.continuation.activation.relu
import com.komputation.instructions.continuation.convolution.convolution
import com.komputation.instructions.continuation.dense.dense
import com.komputation.instructions.loss.squaredLoss
import com.komputation.optimization.historical.momentum
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

    CudaNetwork(
        maximumBatchSize,
        lookup(embeddings, 2, 2, embeddingDimension, optimizationStrategy),
        convolution(numberFilters, filterWidth, filterHeight, initializationStrategy, optimizationStrategy),
        relu(),
        dense(numberClasses, Activation.Softmax, initializationStrategy, optimizationStrategy)
    )
        .training(
            inputs,
            targets,
            5_000,
            squaredLoss(),
            printLoss)
        .run()

}
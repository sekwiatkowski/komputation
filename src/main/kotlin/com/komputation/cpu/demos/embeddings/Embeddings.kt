package com.komputation.cpu.demos.embeddings

import com.komputation.cpu.network.network
import com.komputation.demos.embeddings.EmbeddingData
import com.komputation.initialization.initializeColumnVector
import com.komputation.initialization.uniformInitialization
import com.komputation.instructions.continuation.activation.Activation
import com.komputation.instructions.continuation.convolution.maxPooling
import com.komputation.instructions.continuation.dense.dense
import com.komputation.instructions.entry.lookup
import com.komputation.instructions.loss.squaredLoss
import com.komputation.loss.printLoss
import com.komputation.optimization.historical.momentum
import java.util.*

fun main(args: Array<String>) {

    val random = Random(1)

    val maximumBatchSize = 1
    val numberEmbeddings = 40
    val length = 2
    val embeddingDimension = 2

    val initializationStrategy = uniformInitialization(random, -0.05f, 0.05f)

    val initializeEmbedding = { initializeColumnVector(initializationStrategy, embeddingDimension) }
    val embeddings = Array(numberEmbeddings) { initializeEmbedding() }

    val optimizationStrategy = momentum(0.01f, 0.9f)

    val inputs = EmbeddingData.inputs
    val targets = EmbeddingData.targets
    val numberClasses = EmbeddingData.numberClasses

    network(
        maximumBatchSize,
        lookup(embeddings, length, length, embeddingDimension, optimizationStrategy),
        maxPooling(),
        dense(numberClasses, Activation.Softmax, initializationStrategy, optimizationStrategy)
    )
        .training(
            inputs,
            targets,
            1_000,
            squaredLoss(),
            printLoss)
        .run()

}
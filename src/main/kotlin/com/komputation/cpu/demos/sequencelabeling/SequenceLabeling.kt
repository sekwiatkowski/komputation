package com.komputation.cpu.demos.sequencelabeling

import com.komputation.cpu.layers.recurrent.Direction
import com.komputation.cpu.network.network
import com.komputation.demos.sequencelabeling.SequenceLabelingData
import com.komputation.initialization.initializeColumnVector
import com.komputation.initialization.uniformInitialization
import com.komputation.instructions.continuation.activation.Activation
import com.komputation.instructions.continuation.activation.softmax
import com.komputation.instructions.entry.lookup
import com.komputation.instructions.loss.crossEntropyLoss
import com.komputation.instructions.recurrent.ResultExtraction
import com.komputation.instructions.recurrent.recurrent
import com.komputation.loss.printLoss
import com.komputation.optimization.stochasticGradientDescent
import java.util.*

fun main(args: Array<String>) {

    val random = Random(1)

    val initialization = uniformInitialization(random, -0.01f, 0.01f)
    val optimization = stochasticGradientDescent(0.1f)

    val embeddingDimension = 3

    val input = SequenceLabelingData.input
    val targets = SequenceLabelingData.targets
    val numberCategories = SequenceLabelingData.numberCategories
    val numberSteps = SequenceLabelingData.numberSteps
    val vocabularySize = SequenceLabelingData.vocabularySize

    val initializeEmbedding = { initializeColumnVector(initialization, embeddingDimension) }
    val embeddings = Array(vocabularySize) { initializeEmbedding() }

    network(
            1,
            lookup(embeddings, numberSteps, numberSteps, embeddingDimension, optimization),
            recurrent(
                numberCategories,
                Activation.ReLU,
                ResultExtraction.AllSteps,
                Direction.LeftToRight,
                initialization,
                initialization,
                null,
                optimization),
            softmax()
        )
        .training(
            input,
            targets,
            100,
            crossEntropyLoss(),
            printLoss
        )
        .run()

}
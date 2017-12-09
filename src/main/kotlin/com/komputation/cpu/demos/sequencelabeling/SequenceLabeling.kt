package com.komputation.cpu.demos.sequencelabeling

import com.komputation.cpu.network.Network
import com.komputation.demos.sequencelabeling.SequenceLabelingData
import com.komputation.initialization.initializeColumnVector
import com.komputation.initialization.uniformInitialization
import com.komputation.layers.entry.lookupLayer
import com.komputation.layers.forward.activation.ActivationFunction
import com.komputation.layers.forward.activation.softmaxLayer
import com.komputation.layers.forward.recurrent.recurrentLayer
import com.komputation.loss.crossEntropyLoss
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

    Network(
            1,
            lookupLayer(embeddings, numberSteps, true, embeddingDimension, optimization),
            recurrentLayer(numberSteps, true, embeddingDimension, 3, initialization, null, ActivationFunction.ReLU, optimization),
            softmaxLayer(numberCategories, numberSteps)
        )
        .training(
            input,
            targets,
            100,
            crossEntropyLoss(numberCategories, numberSteps),
            printLoss
        )
        .run()

}
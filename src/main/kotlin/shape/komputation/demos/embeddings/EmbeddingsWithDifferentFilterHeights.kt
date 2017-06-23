package shape.komputation.demos.embeddings

import shape.komputation.initialization.createUniformInitializer
import shape.komputation.initialization.initializeColumnVector
import shape.komputation.layers.feedforward.activation.ReluLayer
import shape.komputation.layers.feedforward.*
import shape.komputation.layers.feedforward.convolution.MaxPoolingLayer
import shape.komputation.layers.feedforward.convolution.createConvolutionalLayer
import shape.komputation.layers.entry.createLookupLayer
import shape.komputation.layers.feedforward.activation.SoftmaxLayer
import shape.komputation.layers.feedforward.projection.createProjectionLayer
import shape.komputation.loss.SquaredLoss
import shape.komputation.matrix.*
import shape.komputation.networks.Network
import shape.komputation.networks.printLoss
import shape.komputation.optimization.momentum
import java.util.*

/*

    bad
    awful
    poor
    abominable
    dreadful
    lousy
    horrible
    unpleasant
    unsatisfactory
    ghastly

    good
    excellent
    virtuous
    great
    satisfactory
    pleasant
    worthy
    beneficial
    splendid
    lovely

    moderately
    pretty
    fairly
    somewhat
    reasonably
    slightly
    mildly
    kind of
    relatively
    sort of

    extremely
    very
    extraordinarily
    exceptionally
    remarkably
    immensely
    unusually
    terribly
    totally
    uncommonly

 */

fun main(args: Array<String>) {

    val random = Random(1)

    val maximumBatchSize = 1
    val numberEmbeddings = 40
    val embeddingDimension = 2

    val initializationStrategy = createUniformInitializer(random, -0.05, 0.05)

    val initializeEmbedding = { initializeColumnVector(initializationStrategy, embeddingDimension) }
    val embeddings = Array(numberEmbeddings) { initializeEmbedding() }

    val numberClasses = 4

    val negativeIndices = 0..9
    val positiveIndices = 10..19
    val weakModifierIndices = 20..29
    val strongModifierIndices = 30..39

    val stronglyNegativeInputs = createInputs(strongModifierIndices, negativeIndices)
    val weaklyNegativeInputs = createInputs(weakModifierIndices, negativeIndices)
    val weaklyPositiveInputs = createInputs(weakModifierIndices, positiveIndices)
    val stronglyPositiveInputs = createInputs(strongModifierIndices, positiveIndices)

    val input = listOf<List<Matrix>>(
        stronglyNegativeInputs,
        weaklyNegativeInputs,
        weaklyPositiveInputs,
        stronglyPositiveInputs

    )
        .flatMap { it }
        .toTypedArray()

    val createTarget = { category : Int -> oneHotVector(numberClasses, category) }

    val targets = listOf(
        (0..9).map { createTarget(0) },
        (0..9).map { createTarget(1) },
        (0..9).map { createTarget(2) },
        (0..9).map { createTarget(3) }
    )
        .flatMap { it }
        .toTypedArray()

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

    val network = Network(
        createLookupLayer(embeddings, embeddingDimension, maximumBatchSize, optimizationStrategy),
        createBranching(
            *filterHeights.map { filterHeight -> createConvolutionSubnetwork(filterHeight) }.toTypedArray()
        ),
        createProjectionLayer(numberFilters * filterHeights.size, numberClasses, true, initializationStrategy, optimizationStrategy),
        SoftmaxLayer()
    )

    network.train(input, targets, SquaredLoss(), 10_000, maximumBatchSize, printLoss)

}

private fun createInputs(modifierIndices: IntRange, polarityIndices: IntRange) =

    modifierIndices.zip(polarityIndices).map { (modifier, polarity) -> intVector(modifier, polarity) }
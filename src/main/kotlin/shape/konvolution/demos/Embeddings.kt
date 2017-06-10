package shape.konvolution.demos

import shape.konvolution.*
import shape.konvolution.layers.continuation.*
import shape.konvolution.layers.continuation.createConvolutionLayer
import shape.konvolution.layers.entry.LookupLayer
import shape.konvolution.loss.SquaredLoss
import shape.konvolution.matrix.*
import shape.konvolution.optimization.StochasticGradientDescent
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

    val numberEmbeddings = 40
    val embeddingDimension = 2

    val generateEntry = createUniformInitializer(random, -0.05, 0.05)
    val initializeEmbedding = { initializeRow(generateEntry, embeddingDimension)}

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

    val createTarget = { category : Int -> createOneHotVector(numberClasses, category) }

    val targets = listOf(
            (0..9).map { createTarget(0) },
            (0..9).map { createTarget(1) },
            (0..9).map { createTarget(2) },
            (0..9).map { createTarget(3) }
        )
        .flatMap { it }
        .toTypedArray()

    val optimizer = StochasticGradientDescent(0.03)

    val numberFilters = 2

    val network = Network(
        LookupLayer(embeddings, optimizer),
        createConvolutionLayer(numberFilters, embeddingDimension, 2, generateEntry, optimizer, optimizer),
        MaxPoolingLayer(),
        ReluLayer(),
        createProjectionLayer(numberFilters, numberClasses, generateEntry, optimizer, optimizer),
        SoftmaxLayer()
    )

    train(network, input, targets, SquaredLoss(), 30_000)

}

private fun createInputs(modifierIndices: IntRange, polarityIndices: IntRange) =
    modifierIndices.zip(polarityIndices).map { (weak, positive) -> createIntegerVector(weak, positive) }
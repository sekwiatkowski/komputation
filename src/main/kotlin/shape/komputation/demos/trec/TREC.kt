package shape.komputation.demos.trec

import shape.komputation.functions.findMaxIndex
import shape.komputation.initialization.uniformInitialization
import shape.komputation.layers.entry.lookupLayer
import shape.komputation.layers.forward.activation.reluLayer
import shape.komputation.layers.forward.activation.softmaxLayer
import shape.komputation.layers.forward.concatenation
import shape.komputation.layers.forward.convolution.convolutionalLayer
import shape.komputation.layers.forward.convolution.maxPoolingLayer
import shape.komputation.layers.forward.dropout.dropoutLayer
import shape.komputation.layers.forward.projection.projectionLayer
import shape.komputation.loss.logisticLoss
import shape.komputation.matrix.IntMatrix
import shape.komputation.matrix.Matrix
import shape.komputation.matrix.intVector
import shape.komputation.matrix.oneHotVector
import shape.komputation.networks.Network
import shape.komputation.optimization.adaptive.rmsprop
import java.io.File
import java.util.*

fun main(args: Array<String>) {

    if (args.size != 2) {

        throw Exception("Please specify the path to the Glove word embeddings and the number of dimensions.")

    }

    val embeddingFilePath = args.first()
    val dimensions = args.last().toInt()

    TrecTraining().run(embeddingFilePath, dimensions)

}

class TrecTraining {

    fun run(embeddingFilePath: String, embeddingDimension: Int) {

        val maximumBatchSize = 1

        val embeddingFile = File(embeddingFilePath)

        val numberFilters = 100
        val filterWidths = intArrayOf(2, 3)
        val filterHeight = embeddingDimension
        val numberFilterWidths = filterWidths.size

        val numberIterations = 30

        val trecDirectory = File(javaClass.classLoader.getResource("trec").toURI())
        val trainingFile = File(trecDirectory, "training.data")
        val testFile = File(trecDirectory, "test.data")

        val trainingExamples = readTrecExamples(trainingFile)
        val testExamples = readTrecExamples(testFile)

        val vocabulary = trainingExamples
            .map { (_, text) -> text }
            .flatMap { tokens -> tokens }
            .toSet()

        val embeddingMap = embedVocabulary(vocabulary, embeddingFile)

        val embeddableVocabulary = embeddingMap.keys.sorted()

        val maximumFilterWidth = filterWidths.max()!!

        val embeddableTrainingExamples = filterExamples(trainingExamples, embeddableVocabulary, maximumFilterWidth)
        val embeddableTestExamples = filterExamples(testExamples, embeddableVocabulary, maximumFilterWidth)

        val trainingRepresentations = represent(embeddableTrainingExamples, embeddableVocabulary)
        val testRepresentations = represent(embeddableTestExamples, embeddableVocabulary)

        val maximumLength = trainingRepresentations.plus(testRepresentations)
            .map { it as IntMatrix }
            .map { it.numberRows }
            .max()!!

        val trainingCategories = embeddableTrainingExamples
            .map { ex -> ex.category }

        val testCategories = embeddableTestExamples
            .map { ex -> ex.category }

        val indexedCategories = trainingCategories
            .toSet()
            .sorted()
            .mapIndexed { index, category -> category to index }
            .toMap()

        val numberCategories = indexedCategories.size

        val trainingTargets = createTargets(trainingCategories, indexedCategories)
        val testTargets = createTargets(testCategories, indexedCategories)
        val numberTestExamples = testTargets.size

        val missing = vocabulary.minus(embeddingMap.keys)

        val embeddings = embeddableVocabulary
            .map { token -> embeddingMap[token]!! }
            .toTypedArray()

        val random = Random(1)
        val initializationStrategy = uniformInitialization(random, -0.05, 0.05)

        val optimizationStrategy = rmsprop(0.001)

        val network = Network(
            lookupLayer(embeddings, embeddingDimension, maximumBatchSize, maximumLength, optimizationStrategy),
            concatenation(
                *filterWidths
                    .map { filterWidth ->
                        arrayOf(
                            convolutionalLayer(numberFilters, filterWidth, filterHeight, initializationStrategy, optimizationStrategy),
                            maxPoolingLayer(),
                            dropoutLayer(numberFilters, random, 0.8, reluLayer())
                        )
                    }
                    .toTypedArray()
            ),
            projectionLayer(numberFilterWidths * numberFilters, numberCategories, initializationStrategy, initializationStrategy, optimizationStrategy),
            softmaxLayer()
        )

        val afterEachIteration = { _ : Int, _ : Double ->

            val accuracyRate = network
                .test(
                    testRepresentations,
                    testTargets,
                    { prediction, target ->

                        findMaxIndex(prediction.entries) == findMaxIndex(target.entries)

                    }
                )
                .count { correct -> correct }
                .div(numberTestExamples.toDouble())

            println(accuracyRate)

        }

        network.train(trainingRepresentations, trainingTargets, logisticLoss(), numberIterations, maximumBatchSize, afterEachIteration)

    }

}

private fun readTrecExamples(source : File) =

    source
        .readLines(Charsets.ISO_8859_1)
        .map { line ->

            val split = line.split(' ')

            val category = split.first().split(":").first()
            val text = split.drop(1).dropLast(1).map { it.toLowerCase() }

            TrecExample(category, text)

        }

data class TrecExample(val category : String, val text : List<String>)

private fun embedVocabulary(vocabulary: Set<String>, embeddingFile: File): Map<String, DoubleArray> {

    val embeddingMap = hashMapOf<String, DoubleArray>()

    embeddingFile.bufferedReader().use { reader ->

        reader
            .lineSequence()
            .forEach { line ->

                val split = line.split(" ")

                val word = split.first()

                if (vocabulary.contains(word)) {

                    val embedding = split.drop(1).map { it.toDouble() }.toDoubleArray()

                    embeddingMap.put(word, embedding)

                }

            }

        embeddingMap

    }

    return embeddingMap

}

private fun filterExamples(examples: Iterable<TrecExample>, vocabulary: Collection<String>, minLength : Int) =

    examples
        .map { (category, text) -> TrecExample(category, text.filter { vocabulary.contains(it) }) }
        .filter { (_, tokens) -> tokens.size >= minLength }


private fun represent(examples: Iterable<TrecExample>, vocabulary: Collection<String>) =

    examples
        .map { (_, tokens) -> tokens }
        .map { tokens -> tokens.map { vocabulary.indexOf(it) }.toIntArray() }
        .map { indices -> intVector(*indices) as Matrix }
        .toTypedArray()

private fun createTargets(trainingCategories: List<String>, indexedCategories: Map<String, Int>) =

    trainingCategories
        .map { category -> oneHotVector(indexedCategories.size, indexedCategories[category]!!) }
        .toTypedArray()
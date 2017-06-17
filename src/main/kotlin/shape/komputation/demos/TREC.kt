package shape.komputation.demos

import shape.komputation.functions.findMaxIndicesInColumns
import shape.komputation.initialization.createUniformInitializer
import shape.komputation.layers.feedforward.activation.ReluLayer
import shape.komputation.layers.feedforward.activation.SoftmaxLayer
import shape.komputation.layers.feedforward.*
import shape.komputation.layers.feedforward.convolution.MaxPoolingLayer
import shape.komputation.layers.feedforward.convolution.createConvolutionalLayer
import shape.komputation.layers.entry.createLookupLayer
import shape.komputation.layers.feedforward.projection.createProjectionLayer
import shape.komputation.loss.LogisticLoss
import shape.komputation.matrix.Matrix
import shape.komputation.matrix.intVector
import shape.komputation.matrix.oneHotVector
import shape.komputation.networks.Network
import shape.komputation.optimization.momentum
import java.io.File
import java.util.*

fun main(args: Array<String>) {

    val embeddingFilePath = args.first()
    val dimensions = args.last().toInt()

    TrecTraining().run(embeddingFilePath, dimensions)

}

class TrecTraining {

    fun run(embeddingFilePath: String, embeddingDimension: Int) {

        val maximumBatchSize = 10

        val embeddingFile = File(embeddingFilePath)

        val numberFilters = 100
        val filterWidth = embeddingDimension
        val filterHeights = intArrayOf(3)
        val numberFilterHeights = filterHeights.size

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

        val maximumFilterHeight = filterHeights.max()!!

        val embeddableTrainingExamples = filterExamles(trainingExamples, embeddableVocabulary, maximumFilterHeight)
        val embeddableTestExamples = filterExamles(testExamples, embeddableVocabulary, maximumFilterHeight)

        val trainingRepresentations = represent(embeddableTrainingExamples, embeddableVocabulary)
        val testRepresentations = represent(embeddableTestExamples, embeddableVocabulary)

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

        val missing = vocabulary.minus(embeddingMap.keys)

        val embeddings = embeddableVocabulary
            .map { token -> embeddingMap[token]!! }
            .toTypedArray()

        val random = Random(1)
        val initializationStrategy = createUniformInitializer(random, -0.05, 0.05)

        val optimizationStrategy = momentum(0.01, 0.1)

        val network = Network(
            createLookupLayer(embeddings, embeddingDimension, maximumBatchSize, optimizationStrategy),
            createConcatenation(
                *filterHeights
                    .map { filterHeight ->

                        arrayOf(
                            createConvolutionalLayer(numberFilters, filterWidth, filterHeight, initializationStrategy, optimizationStrategy),
                            ReluLayer(),
                            MaxPoolingLayer()
                        )
                    }
                    .toTypedArray()
            ),
            createProjectionLayer(numberFilters * numberFilterHeights, numberCategories, true, initializationStrategy, optimizationStrategy),
            SoftmaxLayer()
        )

        val testData = testRepresentations.zip(testTargets)
        val numberTestExamples = testData.size

        network.train(trainingRepresentations, trainingTargets, LogisticLoss(), 10_000, maximumBatchSize) { _, _ ->

            val accuracy = testData
                .count { (input, target) ->

                    val output = network.forward(input)

                    val predictedCategory = findMaxIndicesInColumns(output.entries, output.numberRows, output.numberColumns).first()
                    val actualCategory = findMaxIndicesInColumns(target.entries, target.numberRows, target.numberColumns).first()

                    predictedCategory == actualCategory

                }
                .toDouble()
                .div(numberTestExamples.toDouble())

            println(accuracy)

        }

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

private fun filterExamles(examples: Iterable<TrecExample>, vocabulary: Collection<String>, minLength : Int) =

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
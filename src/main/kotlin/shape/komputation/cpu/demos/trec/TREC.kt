package shape.komputation.cpu.demos.trec

import shape.komputation.cpu.Network
import shape.komputation.cpu.functions.findMaxIndex
import shape.komputation.demos.trec.NLP
import shape.komputation.demos.trec.TRECData
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

        val random = Random(1)
        val initialization = uniformInitialization(random, -0.05, 0.05)

        val optimization = rmsprop(0.001)

        val batchSize = 1

        val embeddingFile = File(embeddingFilePath)

        val numberFilters = 100
        val filterWidths = intArrayOf(2, 3)
        val maximumFilterWidth = filterWidths.max()!!

        val filterHeight = embeddingDimension
        val numberFilterWidths = filterWidths.size

        val numberIterations = 30

        val trecDirectory = File(javaClass.classLoader.getResource("trec").toURI())
        val trainingFile = File(trecDirectory, "training.data")
        val testFile = File(trecDirectory, "test.data")

        val (trainingCategories, trainingDocuments) = TRECData.readExamples(trainingFile)
        val (testCategories, testDocuments) = TRECData.readExamples(testFile)

        val vocabulary = NLP.generateVocabulary(trainingDocuments)

        val embeddingMap = NLP.embedVocabulary(vocabulary, embeddingFile)
        val embeddableVocabulary = embeddingMap.keys.sorted()
        val missing = vocabulary.minus(embeddingMap.keys)

        val trainingDocumentsWithFilteredTokens = NLP.filterTokens(trainingDocuments, embeddableVocabulary)
        val testDocumentsWithFilteredTokens = NLP.filterTokens(testDocuments, embeddableVocabulary)

        val embeddableTrainingIndices = NLP.filterDocuments(trainingDocumentsWithFilteredTokens, maximumFilterWidth)
        val embeddableTestIndices = NLP.filterDocuments(testDocumentsWithFilteredTokens, maximumFilterWidth)

        val embeddableTrainingDocuments = trainingDocumentsWithFilteredTokens.slice(embeddableTrainingIndices)
        val embeddableTestDocuments = testDocumentsWithFilteredTokens.slice(embeddableTestIndices)

        val trainingRepresentations = NLP.vectorizeDocuments(embeddableTrainingDocuments, embeddableVocabulary)
        val testRepresentations = NLP.vectorizeDocuments(embeddableTestDocuments, embeddableVocabulary)

        val embeddableTrainingCategories = trainingCategories.slice(embeddableTrainingIndices)
        val embeddableTestCategories = testCategories.slice(embeddableTestIndices)

        val indexedCategories = NLP.indexCategories(embeddableTrainingCategories.toSet())
        val numberCategories = indexedCategories.size

        val trainingTargets = NLP.createTargets(embeddableTrainingCategories, indexedCategories)
        val testTargets = NLP.createTargets(embeddableTestCategories, indexedCategories)

        val numberTestExamples = testTargets.size

        val maximumLength = trainingRepresentations.plus(testRepresentations)
            .map { it.numberRows }
            .max()!!

        val embeddings = embeddableVocabulary
            .map { token -> embeddingMap[token]!! }
            .toTypedArray()

        val network = Network(
            lookupLayer(embeddings, embeddingDimension, batchSize, maximumLength, optimization),
            concatenation(
                maximumLength * embeddingDimension,
                *filterWidths
                    .map { filterWidth ->
                        arrayOf(
                            convolutionalLayer(numberFilters, filterWidth, filterHeight, initialization, optimization),
                            maxPoolingLayer(),
                            dropoutLayer(numberFilters, random, 0.8, reluLayer(numberFilters))
                        )
                    }
                    .toTypedArray()
            ),
            projectionLayer(numberFilterWidths * numberFilters, numberCategories, initialization, initialization, optimization),
            softmaxLayer(numberCategories)
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

        network.train(trainingRepresentations, trainingTargets, logisticLoss(numberCategories), numberIterations, batchSize, afterEachIteration)

    }

}
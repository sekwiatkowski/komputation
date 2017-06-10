package shape.konvolution.demos

import shape.konvolution.matrix.createRealMatrix
import shape.konvolution.matrix.createOneHotVector
import java.io.File

fun main(args: Array<String>) {

    Test().run()


}

class Test {

    fun run() {

        val trainingFile = File(javaClass.classLoader.getResource("train_5500.label").toURI())

        val trainingExamples = readTrecExamples(trainingFile)

        val indexedCategories = trainingExamples
            .map { ex -> ex.category }
            .toSet()
            .sorted()
            .mapIndexed { index, category -> category to index }
            .toMap()

        val numberCategories = indexedCategories.size

        val targets = trainingExamples
            .map { (category, _) -> category }
            .map { category -> createOneHotVector(numberCategories, indexedCategories[category]!!) }

        val vocabulary = trainingExamples
            .map { (_, text) -> text }
            .flatMap { tokens -> tokens }
            .toSet()

        val embeddingDimension = 300

        val embeddingFile = File("C:\\Users\\Sebastian\\Documents\\Glove\\glove.6B.${embeddingDimension}d.txt")

        val embeddingMap = embeddingFile.bufferedReader().use { reader ->


            reader
                .lineSequence()
                .map { line ->

                    val split = line.split(" ")

                    val word = split.first()
                    val embedding = split.drop(1).map { it.toDouble() }.toDoubleArray()

                    word to embedding

                }
                .filter { (word, _) ->
                    vocabulary.contains(word)
                }
                .toMap()

        }

        val missing = vocabulary.minus(embeddingMap.keys)

        val finalVocabulary = embeddingMap.keys.sorted()

        val embeddings = finalVocabulary.map { token -> embeddingMap[token]!! }

        val representations = trainingExamples
            .map { (_, text) ->

                text.filter { finalVocabulary.contains(it) }
            }
            .map { tokens ->

                val rows = tokens.map { doubleArrayOf(finalVocabulary.indexOf(it).toDouble()) }.toTypedArray()

                val matrix = createRealMatrix(*rows)

                matrix
            }

    }

    private fun readTrecExamples(source : File) =

        source
            .readLines(Charsets.ISO_8859_1)
            .map { line ->

                val split = line.split(' ')

                val category = split.first().split(":").first()
                val text = split.drop(1).map { it.toLowerCase() }

                TrecExample(category, text)

            }

}

data class TrecExample(val category : String, val text : List<String>)

data class WordEmbedding(val word : String, val embedding : DoubleArray)
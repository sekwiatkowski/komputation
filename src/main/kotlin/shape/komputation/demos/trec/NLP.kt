package shape.komputation.demos.trec

import shape.komputation.matrix.Matrix
import shape.komputation.matrix.intColumnVector
import shape.komputation.matrix.oneHotVector
import java.io.File

object NLP {

    fun generateVocabulary(documents: Iterable<List<String>>) =

        documents
            .flatMap { tokens -> tokens }
            .toSet()

    fun embedVocabulary(vocabulary: Set<String>, embeddingFile: File): Map<String, DoubleArray> {

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

    fun filterDocuments(documents: Iterable<List<String>>, vocabulary: Collection<String>, minLength : Int) =

        documents
            .map { document -> document.filter { vocabulary.contains(it) } }
            .filter { document -> document.size >= minLength }

    fun vectorizeDocuments(documents: Iterable<List<String>>, vocabulary: Collection<String>) =

        documents
            .map { tokens -> tokens.map { vocabulary.indexOf(it) }.toIntArray() }
            .map { indices -> intColumnVector(*indices) as Matrix }
            .toTypedArray()

    fun indexCategories(categories: Set<String>) =

        categories
            .toSet()
            .sorted()
            .mapIndexed { index, category -> category to index }
            .toMap()

    fun createTargets(categories: Iterable<String>, indexedCategories: Map<String, Int>) =

        categories
            .map { category -> oneHotVector(indexedCategories.size, indexedCategories[category]!!) }
            .toTypedArray()

}
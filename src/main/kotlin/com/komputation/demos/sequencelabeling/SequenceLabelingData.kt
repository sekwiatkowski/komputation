package com.komputation.demos.sequencelabeling

import com.komputation.matrix.floatArrayFromColumns
import com.komputation.matrix.intMatrix
import com.komputation.matrix.oneHotArray

object SequenceLabelingData {

    private val sentences =
        listOf(
            listOf("i", "am", "."),
            listOf("you", "are", "."),
            listOf("he", "is", "."),
            listOf("he", "is", "."),
            listOf("it", "is", "."),
            listOf("we", "are", "."),
            listOf("you", "are", "."),
            listOf("they", "are", "."),
            listOf("am", "I", "?"),
            listOf("are", "you", "?"),
            listOf("is", "he", "?"),
            listOf("is", "she", "?"),
            listOf("is", "it", "?"),
            listOf("are", "we", "?"),
            listOf("are", "you", "?"),
            listOf("are", "they", "?")
        )

    private val vocabulary = this.sentences.flatMap { sentence -> sentence }.toSet().toList()
    val vocabularySize = vocabulary.size

    val input = this.sentences
        .map { sentence ->
            intMatrix(*sentence.map { token -> this.vocabulary.indexOf(token) }.toIntArray())
        }
        .toTypedArray()

    private val labels = listOf(
        listOf(0, 1, 2),
        listOf(0, 1, 2),
        listOf(0, 1, 2),
        listOf(0, 1, 2),
        listOf(0, 1, 2),
        listOf(0, 1, 2),
        listOf(0, 1, 2),
        listOf(0, 1, 2),
        listOf(1, 0, 2),
        listOf(1, 0, 2),
        listOf(1, 0, 2),
        listOf(1, 0, 2),
        listOf(1, 0, 2),
        listOf(1, 0, 2),
        listOf(1, 0, 2),
        listOf(1, 0, 2)
    )

    private fun represent(label : List<Int>): FloatArray {
        val columns = Array(3) { index ->
            oneHotArray(3, label[index], 1.0f)
        }

        return floatArrayFromColumns(3, 3, *columns)
    }

    val targets = this.labels
        .map { label ->
            represent(label)
        }
        .toTypedArray()

    const val numberCategories = 3
    const val numberSteps = 3
}
package com.komputation.demos.mnist

import com.komputation.matrix.FloatMatrix
import com.komputation.matrix.Matrix
import com.komputation.matrix.oneHotArray
import java.io.File

object MnistData {

    val numberCategories = 10

    private fun loadMnist(csvFile: File): Pair<Array<Matrix>, Array<FloatArray>> {
        val inputs = arrayListOf<Matrix>()
        val targets = arrayListOf<FloatArray>()

        csvFile
            .bufferedReader()
            .lineSequence()
            .forEachIndexed { index, line ->

                val split = line
                    .split(",")

                val category = split.first().toInt()

                val input = FloatMatrix(split.drop(1).map { it.toFloat().div(255.0f) }.toFloatArray(), 28, 28)
                val target = oneHotArray(this.numberCategories, category)

                inputs.add(input)
                targets.add(target)
            }

        return inputs.toTypedArray() to targets.toTypedArray()
    }

    fun loadMnistTraining(csvFile: File) =
        loadMnist(csvFile)

    fun loadMnistTest(csvFile: File) =
        loadMnist(csvFile)

}
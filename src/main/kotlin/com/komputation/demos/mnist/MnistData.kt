package com.komputation.demos.mnist

import com.komputation.matrix.FloatMatrix
import com.komputation.matrix.oneHotArray
import java.io.File

object MnistData {

    const val numberCategories = 10

    private fun loadMnist(csvFile: File, useOneDimension : Boolean): Pair<Array<FloatMatrix>, Array<FloatArray>> {
        val inputs = arrayListOf<FloatMatrix>()
        val targets = arrayListOf<FloatArray>()

        val numberRows = if(useOneDimension) 784 else 28
        val numberColumns = if(useOneDimension) 1 else 28

        csvFile
            .bufferedReader()
            .lineSequence()
            .forEach { line ->

                val split = line
                    .split(",")

                val category = split.first().toInt()

                val input = FloatMatrix(split.drop(1).map { it.toFloat().div(255.0f) }.toFloatArray(), numberRows, numberColumns)
                val target = oneHotArray(this.numberCategories, category)

                inputs.add(input)
                targets.add(target)
            }

        return inputs.toTypedArray() to targets.toTypedArray()
    }

    fun loadMnistTraining(csvFile: File, useOneDimension : Boolean) =
        loadMnist(csvFile, useOneDimension)

    fun loadMnistTest(csvFile: File, useOneDimension : Boolean) =
        loadMnist(csvFile, useOneDimension)

}
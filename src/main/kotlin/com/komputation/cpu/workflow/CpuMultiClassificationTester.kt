package com.komputation.cpu.workflow

import com.komputation.cpu.functions.findMaxIndices

class CpuMulticlassTester(
    private val numberCategories: Int,
    private val length : Int) : CpuClassificationTester {

    private val actualCategories = IntArray(this.length)
    private val predictedCategories = IntArray(this.length)

    override fun test(predictions : FloatArray, targets : FloatArray): Boolean {

        var isCorrect = true
        var step = 0

        while(step < this.length) {

            findMaxIndices(targets, this.numberCategories, this.length, this.actualCategories)
            findMaxIndices(predictions, this.numberCategories, this.length, this.predictedCategories)

            isCorrect = this.actualCategories.contentEquals(this.predictedCategories)

            if (isCorrect) {

                step++

            }
            else {

                break

            }

        }

        return isCorrect

    }

}
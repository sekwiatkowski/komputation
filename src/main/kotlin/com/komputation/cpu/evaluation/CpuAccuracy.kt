package com.komputation.cpu.evaluation

fun computeAccuracy(predictions: Array<IntArray>, targets: Array<IntArray>, numberComparisons : Int): Float {

    var correctPredictions = 0

    for (indexComparison in 0 until numberComparisons) {

        if(targets[indexComparison].contentEquals(predictions[indexComparison])) {

            correctPredictions++

        }

    }

    return correctPredictions.toFloat().div(numberComparisons.toFloat())

}
package com.komputation.initialization

fun initializeColumnVector(strategy: InitializationStrategy, numberRows: Int) =
    initializeWeights(strategy, numberRows,1, numberRows)

fun initializeWeights(strategy: InitializationStrategy, numberRows: Int, numberColumns : Int, numberIncoming : Int): FloatArray {
    val array = FloatArray(numberRows * numberColumns)

    for (indexRow in 0 until numberRows) {
        for (indexColumn in 0 until numberColumns) {
            array[indexColumn * numberRows + indexRow] = strategy.initialize(indexRow, indexColumn, numberIncoming)
        }
    }

    return array
}
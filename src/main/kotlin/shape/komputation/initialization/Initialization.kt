package shape.komputation.initialization

fun initializeColumnVector(strategy: InitializationStrategy, numberRows: Int): DoubleArray {

    return initializeMatrix(strategy, numberRows,1, numberRows)

}

fun initializeMatrix(strategy: InitializationStrategy, numberRows: Int, numberColumns : Int, numberIncoming : Int): DoubleArray {

    val array = DoubleArray(numberRows * numberColumns)

    for (indexRow in 0..numberRows - 1) {

        for (indexColumn in 0..numberColumns - 1) {

            array[indexColumn * numberRows + indexRow] = strategy.initialize(indexRow, indexColumn, numberIncoming)

        }

    }

    return array

}
package shape.komputation.initialization

fun initializeColumnVector(strategy: InitializationStrategy, numberColumns: Int): DoubleArray {

    return initializeMatrix(strategy, 1, numberColumns)

}

fun initializeRowVector(strategy: InitializationStrategy, numberColumns: Int): DoubleArray {

    return initializeMatrix(strategy, numberColumns, 1)

}

fun initializeMatrix(strategy: InitializationStrategy, numberRows: Int, numberColumns : Int): DoubleArray {

    val array = DoubleArray(numberRows * numberColumns)

    for (indexRow in 0..numberRows - 1) {

        for (indexColumn in 0..numberColumns - 1) {

            array[indexColumn * numberRows + indexRow] = strategy(indexRow, indexColumn)

        }

    }

    return array

}
package shape.komputation.initialization

import shape.komputation.matrix.createRealMatrix


fun initializeMatrix(strategy : InitializationStrategy, numberRows : Int, numberColumns: Int) =

    Array(numberRows) { indexRow ->

        initializeRow(strategy, indexRow, numberColumns)

    }
    .let { rows ->

        createRealMatrix(*rows)
    }

fun initializeRow(strategy: InitializationStrategy, indexRow: Int, numberColumns: Int) =

    DoubleArray(numberColumns) { indexColumn ->

        strategy(indexRow, indexColumn)

    }

fun initializeRowVector(strategy : InitializationStrategy, numberRows: Int) =

    initializeMatrix(strategy, numberRows, 1)

fun initializeColumnVector(strategy : InitializationStrategy, numberColumns: Int) =

    initializeMatrix(strategy, 1, numberColumns)
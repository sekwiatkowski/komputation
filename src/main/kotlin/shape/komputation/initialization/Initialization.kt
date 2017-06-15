package shape.komputation.initialization

import shape.komputation.matrix.createRealMatrix


fun initializeMatrix(strategy : InitializationStrategy, numberRows : Int, numberColumns: Int) =

    Array(numberRows) { indexRow ->

        initializeRow(strategy, numberColumns)

    }
    .let { rows ->

        createRealMatrix(*rows)
    }

fun initializeRow(strategy: InitializationStrategy, numberColumns: Int) =

    DoubleArray(numberColumns) { indexColumn ->

        strategy(0, indexColumn)

    }

fun initializeColumn(strategy: InitializationStrategy, numberRows: Int) =

    DoubleArray(numberRows) { indexRow ->

        strategy(indexRow, 0)

    }


fun initializeRowVector(strategy : InitializationStrategy, numberRows: Int) =

    initializeMatrix(strategy, numberRows, 1)

fun initializeColumnVector(strategy : InitializationStrategy, numberColumns: Int) =

    initializeMatrix(strategy, 1, numberColumns)
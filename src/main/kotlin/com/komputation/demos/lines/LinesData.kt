package com.komputation.demos.lines

import com.komputation.matrix.Matrix
import com.komputation.matrix.floatMatrixFromRows

object LinesData {

    private val numberRows = 3
    private val numberColumns = 3

    val inputs = arrayOf<Matrix>(
        floatMatrixFromRows(
            this.numberRows,
            this.numberColumns,
            floatArrayOf(1.0f, 1.0f, 1.0f),
            floatArrayOf(0.0f, 0.0f, 0.0f),
            floatArrayOf(0.0f, 0.0f, 0.0f)
        ),
        floatMatrixFromRows(
            this.numberRows,
            this.numberColumns,
            floatArrayOf(0.0f, 0.0f, 0.0f),
            floatArrayOf(1.0f, 1.0f, 1.0f),
            floatArrayOf(0.0f, 0.0f, 0.0f)
        ),
        floatMatrixFromRows(
            this.numberRows,
            this.numberColumns,
            floatArrayOf(0.0f, 0.0f, 0.0f),
            floatArrayOf(0.0f, 0.0f, 0.0f),
            floatArrayOf(1.0f, 1.0f, 1.0f)
        ),
        floatMatrixFromRows(
            this.numberRows,
            this.numberColumns,
            floatArrayOf(1.0f, 0.0f, 0.0f),
            floatArrayOf(1.0f, 0.0f, 0.0f),
            floatArrayOf(1.0f, 0.0f, 0.0f)
        ),
        floatMatrixFromRows(
            this.numberRows,
            this.numberColumns,
            floatArrayOf(0.0f, 1.0f, 0.0f),
            floatArrayOf(0.0f, 1.0f, 0.0f),
            floatArrayOf(0.0f, 1.0f, 0.0f)
        ),
        floatMatrixFromRows(
            this.numberRows,
            this.numberColumns,
            floatArrayOf(0.0f, 0.0f, 1.0f),
            floatArrayOf(0.0f, 0.0f, 1.0f),
            floatArrayOf(0.0f, 0.0f, 1.0f)
        )
    )

    val targets = arrayOf(
        floatArrayOf(0.0f, 1.0f),
        floatArrayOf(0.0f, 1.0f),
        floatArrayOf(0.0f, 1.0f),
        floatArrayOf(1.0f, 0.0f),
        floatArrayOf(1.0f, 0.0f),
        floatArrayOf(1.0f, 0.0f)
    )

}
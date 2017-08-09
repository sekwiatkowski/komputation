package shape.komputation.demos.lines

import shape.komputation.matrix.Matrix
import shape.komputation.matrix.floatMatrixFromRows

object LinesData {

    val inputs = arrayOf<Matrix>(

        floatMatrixFromRows(
            floatArrayOf(1.0f, 1.0f, 1.0f),
            floatArrayOf(0.0f, 0.0f, 0.0f),
            floatArrayOf(0.0f, 0.0f, 0.0f)
        ),
        floatMatrixFromRows(
            floatArrayOf(0.0f, 0.0f, 0.0f),
            floatArrayOf(1.0f, 1.0f, 1.0f),
            floatArrayOf(0.0f, 0.0f, 0.0f)
        ),
        floatMatrixFromRows(
            floatArrayOf(0.0f, 0.0f, 0.0f),
            floatArrayOf(0.0f, 0.0f, 0.0f),
            floatArrayOf(1.0f, 1.0f, 1.0f)
        ),
        floatMatrixFromRows(
            floatArrayOf(1.0f, 0.0f, 0.0f),
            floatArrayOf(1.0f, 0.0f, 0.0f),
            floatArrayOf(1.0f, 0.0f, 0.0f)
        ),
        floatMatrixFromRows(
            floatArrayOf(0.0f, 1.0f, 0.0f),
            floatArrayOf(0.0f, 1.0f, 0.0f),
            floatArrayOf(0.0f, 1.0f, 0.0f)
        ),
        floatMatrixFromRows(
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
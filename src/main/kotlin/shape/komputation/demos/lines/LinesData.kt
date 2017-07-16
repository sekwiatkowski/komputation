package shape.komputation.demos.lines

import shape.komputation.matrix.Matrix
import shape.komputation.matrix.doubleColumnVector
import shape.komputation.matrix.doubleMatrixFromRows

object LinesData {

    val inputs = arrayOf<Matrix>(

        doubleMatrixFromRows(
            doubleArrayOf(1.0, 1.0, 1.0),
            doubleArrayOf(0.0, 0.0, 0.0),
            doubleArrayOf(0.0, 0.0, 0.0)
        ),
        doubleMatrixFromRows(
            doubleArrayOf(0.0, 0.0, 0.0),
            doubleArrayOf(1.0, 1.0, 1.0),
            doubleArrayOf(0.0, 0.0, 0.0)
        ),
        doubleMatrixFromRows(
            doubleArrayOf(0.0, 0.0, 0.0),
            doubleArrayOf(0.0, 0.0, 0.0),
            doubleArrayOf(1.0, 1.0, 1.0)
        ),
        doubleMatrixFromRows(
            doubleArrayOf(1.0, 0.0, 0.0),
            doubleArrayOf(1.0, 0.0, 0.0),
            doubleArrayOf(1.0, 0.0, 0.0)
        ),
        doubleMatrixFromRows(
            doubleArrayOf(0.0, 1.0, 0.0),
            doubleArrayOf(0.0, 1.0, 0.0),
            doubleArrayOf(0.0, 1.0, 0.0)
        ),
        doubleMatrixFromRows(
            doubleArrayOf(0.0, 0.0, 1.0),
            doubleArrayOf(0.0, 0.0, 1.0),
            doubleArrayOf(0.0, 0.0, 1.0)
        )
    )

    val targets = arrayOf(
        doubleColumnVector(0.0, 1.0),
        doubleColumnVector(0.0, 1.0),
        doubleColumnVector(0.0, 1.0),
        doubleColumnVector(1.0, 0.0),
        doubleColumnVector(1.0, 0.0),
        doubleColumnVector(1.0, 0.0)
    )

}
package shape.komputation.demos.and

import shape.komputation.matrix.Matrix
import shape.komputation.matrix.doubleColumnVector

object OneHotAndData {

    val input = arrayOf<Matrix>(
        doubleColumnVector(0.0, 0.0),
        doubleColumnVector(0.0, 1.0),
        doubleColumnVector(1.0, 0.0),
        doubleColumnVector(1.0, 1.0)
    )

    val targets = arrayOf(
        doubleColumnVector(1.0, 0.0),
        doubleColumnVector(1.0, 0.0),
        doubleColumnVector(1.0, 0.0),
        doubleColumnVector(0.0, 1.0)
    )


}
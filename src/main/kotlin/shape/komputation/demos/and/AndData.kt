package shape.komputation.demos.and

import shape.komputation.matrix.Matrix
import shape.komputation.matrix.doubleColumnVector
import shape.komputation.matrix.doubleScalar

object AndData {

    val inputs = arrayOf<Matrix>(
        doubleColumnVector(0.0, 0.0),
        doubleColumnVector(1.0, 0.0),
        doubleColumnVector(0.0, 1.0),
        doubleColumnVector(1.0, 1.0))

    val targets = arrayOf(
        doubleScalar(0.0),
        doubleScalar(0.0),
        doubleScalar(0.0),
        doubleScalar(1.0)
    )

}
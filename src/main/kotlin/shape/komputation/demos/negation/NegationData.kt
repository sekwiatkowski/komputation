package shape.komputation.demos.negation

import shape.komputation.matrix.Matrix
import shape.komputation.matrix.floatScalar

object NegationData {

    val inputs = arrayOf<Matrix>(
        floatScalar(0.0f),
        floatScalar(1.0f)
    )

    val targets = arrayOf(
        floatScalar(1.0f),
        floatScalar(0.0f)
    )

}
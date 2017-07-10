package shape.komputation.demos.negation

import shape.komputation.matrix.Matrix
import shape.komputation.matrix.doubleScalar

object NegationData {

    val inputs = arrayOf<Matrix>(
        doubleScalar(0.0),
        doubleScalar(1.0)
    )

    val targets = arrayOf(
        doubleScalar(1.0),
        doubleScalar(0.0)
    )

}
package shape.komputation.demos.negation

import shape.komputation.matrix.Matrix
import shape.komputation.matrix.floatMatrix

object NegationData {

    val inputs = arrayOf<Matrix>(
        floatMatrix(0.0f),
        floatMatrix(1.0f)
    )

    val targets = arrayOf(
        floatArrayOf(1.0f),
        floatArrayOf(0.0f)
    )

}
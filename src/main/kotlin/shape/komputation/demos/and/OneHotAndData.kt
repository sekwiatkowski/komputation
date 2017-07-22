package shape.komputation.demos.and

import shape.komputation.matrix.Matrix
import shape.komputation.matrix.floatColumnVector

object OneHotAndData {

    val input = arrayOf<Matrix>(
        floatColumnVector(0.0f, 0.0f),
        floatColumnVector(0.0f, 1.0f),
        floatColumnVector(1.0f, 0.0f),
        floatColumnVector(1.0f, 1.0f)
    )

    val targets = arrayOf(
        floatColumnVector(1.0f, 0.0f),
        floatColumnVector(1.0f, 0.0f),
        floatColumnVector(1.0f, 0.0f),
        floatColumnVector(0.0f, 1.0f)
    )


}
package shape.komputation.demos.and

import shape.komputation.matrix.Matrix
import shape.komputation.matrix.floatColumnVector
import shape.komputation.matrix.floatScalar

object BinaryAndData {

    val inputs = arrayOf<Matrix>(
        floatColumnVector(0.0f, 0.0f),
        floatColumnVector(1.0f, 0.0f),
        floatColumnVector(0.0f, 1.0f),
        floatColumnVector(1.0f, 1.0f))

    val targets = arrayOf(
        floatScalar(0.0f),
        floatScalar(0.0f),
        
        floatScalar(0.0f),
        floatScalar(1.0f)
    )

}
package com.komputation.demos.and

import com.komputation.matrix.Matrix
import com.komputation.matrix.floatMatrix

object BinaryAndData {

    val inputs = arrayOf<Matrix>(
        floatMatrix(2, 1, 0.0f, 0.0f),
        floatMatrix(2, 1, 1.0f, 0.0f),
        floatMatrix(2, 1, 0.0f, 1.0f),
        floatMatrix(2, 1, 1.0f, 1.0f))

    val targets = arrayOf(
        floatArrayOf(0.0f),
        floatArrayOf(0.0f),
        floatArrayOf(0.0f),
        floatArrayOf(1.0f)
    )

}
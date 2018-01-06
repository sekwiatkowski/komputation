package com.komputation.demos.not

import com.komputation.matrix.floatMatrix

object NotData {

    val inputs = arrayOf(
        floatMatrix(1, 1, 0.0f),
        floatMatrix(1, 1, 1.0f)
    )

    val targets = arrayOf(
        floatArrayOf(1.0f),
        floatArrayOf(0.0f)
    )

}
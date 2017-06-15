package shape.komputation.functions

import org.junit.jupiter.api.Test
import shape.komputation.assertMatrixEquality
import shape.komputation.functions.convolution.maxPooling
import shape.komputation.matrix.createRealMatrix

class MaxPoolingTest {

    @Test
    fun test() {

        val input = createRealMatrix(
            doubleArrayOf(1.0, 2.0),
            doubleArrayOf(3.0, -4.0)
        )

        val expected = createRealMatrix(
            doubleArrayOf(2.0),
            doubleArrayOf(3.0)
        )

        val actual = maxPooling(input)

        assertMatrixEquality(expected, actual, 0.001)

    }

}
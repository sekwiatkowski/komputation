package shape.komputation.functions.activation

import org.junit.jupiter.api.Test
import shape.komputation.assertMatrixEquality
import shape.komputation.matrix.createRealMatrix

class SoftmaxTest {

    @Test
    fun test() {

        val input = createRealMatrix(doubleArrayOf(1.0, 2.0), doubleArrayOf(3.0, 4.0))

        val expected = createRealMatrix(
            doubleArrayOf(Math.exp(1.0) / (Math.exp(1.0) + Math.exp(3.0)), Math.exp(2.0) / (Math.exp(2.0) + Math.exp(4.0))),
            doubleArrayOf(Math.exp(3.0) / (Math.exp(1.0) + Math.exp(3.0)), Math.exp(4.0) / (Math.exp(2.0) + Math.exp(4.0)))
        )

        val actual = softmax(input)

        assertMatrixEquality(expected, actual, 0.0001)

    }

}
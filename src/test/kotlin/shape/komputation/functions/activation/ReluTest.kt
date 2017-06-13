package shape.komputation.functions.activation

import org.junit.jupiter.api.Test
import shape.komputation.assertMatrixEquality
import shape.komputation.functions.activation.relu
import shape.komputation.matrix.createRealMatrix

class ReluTest {

    @Test
    fun testForward() {

        val input = createRealMatrix(doubleArrayOf(1.0, -2.0), doubleArrayOf(-3.0, 4.0))

        val expected = createRealMatrix(
            doubleArrayOf(1.0, 0.0),
            doubleArrayOf(0.0, 4.0)
        )

        val actual = relu(input)

        assertMatrixEquality(expected, actual, 0.0001)

    }

}
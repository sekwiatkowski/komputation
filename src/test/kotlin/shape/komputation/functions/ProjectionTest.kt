package shape.komputation.functions

import org.junit.jupiter.api.Test
import shape.komputation.assertMatrixEquality
import shape.komputation.matrix.createRealMatrix

class ProjectionTest {

    @Test
    fun testForward() {

        val weights = createRealMatrix(doubleArrayOf(1.0, 2.0, 3.0), doubleArrayOf(4.0, 5.0, 6.0))

        val input = createRealMatrix(doubleArrayOf(1.0), doubleArrayOf(2.0), doubleArrayOf(3.0))

        val expected = createRealMatrix(
            doubleArrayOf(1.0 * 1.0 + 2.0 * 2.0 + 3.0 * 3.0),
            doubleArrayOf(4.0 * 1.0 + 5.0 * 2.0 + 6.0 * 3.0)
        )

        val actual = project(input, weights, null)

        assertMatrixEquality(expected, actual, 0.001)

    }

}
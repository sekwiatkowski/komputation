package shape.komputation.matrix

import org.junit.jupiter.api.Test
import shape.komputation.assertMatrixEquality

class RealMatrixTest {

    @Test
    fun test() {

        val matrix = createRealMatrix(
            doubleArrayOf(1.1, 2.1),
            doubleArrayOf(1.2, 2.2)
        )

        val actualFirstColumn = matrix.getColumn(0)
        val actualSecondColumn = matrix.getColumn(1)

        val expectedFirstColumn = createRealVector(1.1, 1.2)
        val expectedSecondColumn = createRealVector(2.1, 2.2)

        assertMatrixEquality(expectedFirstColumn, actualFirstColumn, 0.01)
        assertMatrixEquality(expectedSecondColumn, actualSecondColumn, 0.01)

    }

}
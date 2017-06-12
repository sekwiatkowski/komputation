package shape.konvolution.functions

import org.junit.jupiter.api.Test
import shape.konvolution.assertMatrixEquality
import shape.konvolution.matrix.createRealMatrix
import shape.konvolution.matrix.createRealVector

class RowConcatenationTest {

    @Test
    fun test1() {

        val firstVector = createRealVector(1.0, 2.0)
        val secondVector = createRealVector(3.0, 4.0)

        val actual = concatRows(firstVector, secondVector)
        val expected = createRealVector(
            1.0,
            2.0,
            3.0,
            4.0
        )

        assertMatrixEquality(expected, actual, 0.01)

    }

    @Test
    fun test2() {

        val firstVector = createRealMatrix(
            doubleArrayOf(1.0, 2.0)
        )

        val secondVector = createRealMatrix(
            doubleArrayOf(3.0, 4.0),
            doubleArrayOf(5.0, 6.0)
        )

        val actual = concatRows(firstVector, secondVector)
        val expected = createRealMatrix(
            doubleArrayOf(1.0, 2.0),
            doubleArrayOf(3.0, 4.0),
            doubleArrayOf(5.0, 6.0)
        )

        assertMatrixEquality(expected, actual, 0.01)

    }

}
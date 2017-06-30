package shape.komputation.matrix

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test
import shape.komputation.assertMatrixEquality

class DoubleMatrixTest {

    @Test
    fun testGetEntry() {

        val matrix = DoubleMatrix(2, 2, doubleArrayOf(1.0, 2.0, 3.0, 4.0))

        assertEquals(matrix.getEntry(0, 0), 1.0)
        assertEquals(matrix.getEntry(1, 0), 2.0)
        assertEquals(matrix.getEntry(0, 1), 3.0)
        assertEquals(matrix.getEntry(1, 1), 4.0)

    }

    @Test
    fun testSetEntry() {

        val actual = DoubleMatrix(2, 2, doubleArrayOf(1.0, 2.0, 3.0, 4.0))

        actual.setEntry(0, 0, -1.0)
        actual.setEntry(1, 0, -2.0)
        actual.setEntry(0, 1, -3.0)
        actual.setEntry(1, 1, -4.0)

        val expected = DoubleMatrix(2, 2, doubleArrayOf(-1.0, -2.0, -3.0, -4.0))

        assertMatrixEquality(expected, actual, 0.001)

    }

    @Test
    fun testGetColumn() {

        val matrix = DoubleMatrix(2, 2, doubleArrayOf(1.0, 2.0, 3.0, 4.0))

        val actual = matrix.getColumn(0)
        val expected = doubleColumnVector(1.0, 2.0)

        assertMatrixEquality(expected, actual, 0.001)

    }

    @Test
    fun testSetColumn() {

        val actual = DoubleMatrix(2, 2, doubleArrayOf(1.0, 2.0, 3.0, 4.0))

        actual.setColumn(0, doubleArrayOf(-1.0, -2.0))
        actual.setColumn(1, doubleArrayOf(-3.0, -4.0))

        val expected = DoubleMatrix(2, 2, doubleArrayOf(-1.0, -2.0, -3.0, -4.0))

        assertMatrixEquality(expected, actual, 0.001)

    }

}
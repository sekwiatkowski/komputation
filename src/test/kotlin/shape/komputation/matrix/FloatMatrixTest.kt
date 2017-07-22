package shape.komputation.matrix

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test
import shape.komputation.assertMatrixEquality

class FloatMatrixTest {

    @Test
    fun testGetEntry() {

        val matrix = FloatMatrix(2, 2, floatArrayOf(1.0f, 2.0f, 3.0f, 4.0f))

        assertEquals(1.0f, matrix.getEntry(0, 0))
        assertEquals(2.0f, matrix.getEntry(1, 0))
        assertEquals(3.0f, matrix.getEntry(0, 1))
        assertEquals(4.0f, matrix.getEntry(1, 1))

    }

    @Test
    fun testSetEntry() {

        val actual = FloatMatrix(2, 2, floatArrayOf(1.0f, 2.0f, 3.0f, 4.0f))

        actual.setEntry(0, 0, -1.0f)
        actual.setEntry(1, 0, -2.0f)
        actual.setEntry(0, 1, -3.0f)
        actual.setEntry(1, 1, -4.0f)

        val expected = FloatMatrix(2, 2, floatArrayOf(-1.0f, -2.0f, -3.0f, -4.0f))

        assertMatrixEquality(expected, actual, 0.001f)

    }

    @Test
    fun testGetColumn() {

        val matrix = FloatMatrix(2, 2, floatArrayOf(1.0f, 2.0f, 3.0f, 4.0f))

        val actual = matrix.getColumn(0)
        val expected = floatColumnVector(1.0f, 2.0f)

        assertMatrixEquality(expected, actual, 0.001f)

    }

    @Test
    fun testSetColumn() {

        val actual = FloatMatrix(2, 2, floatArrayOf(1.0f, 2.0f, 3.0f, 4.0f))

        actual.setColumn(0, floatArrayOf(-1.0f, -2.0f))
        actual.setColumn(1, floatArrayOf(-3.0f, -4.0f))

        val expected = FloatMatrix(2, 2, floatArrayOf(-1.0f, -2.0f, -3.0f, -4.0f))

        assertMatrixEquality(expected, actual, 0.001f)

    }

}
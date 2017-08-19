package shape.komputation.cpu.functions

import org.junit.Assert.assertArrayEquals
import org.junit.Test

class ConcatenationTest {

    @Test
    fun testOneArrayOneDimension() {

        val expected = floatArrayOf(1f)
        val actual = FloatArray(1)

        denselyConcatenateFloatArrays(arrayOf(floatArrayOf(1f)), 1, actual)

        assertArrayEquals(expected, actual, 0.01f)

    }

    @Test
    fun testOneArrayTwoDimensions() {

        val expected = floatArrayOf(1f, 2f)
        val actual = FloatArray(2)

        denselyConcatenateFloatArrays(arrayOf(floatArrayOf(1f, 2f)), 2, actual)

        assertArrayEquals(expected, actual, 0.01f)

    }

    @Test
    fun testTwoArraysOneDimension() {

        val expected = floatArrayOf(1f, 2f)
        val actual = FloatArray(2)

        denselyConcatenateFloatArrays(arrayOf(floatArrayOf(1f), floatArrayOf(2f)), 1, actual)

        assertArrayEquals(expected, actual, 0.01f)

    }


}
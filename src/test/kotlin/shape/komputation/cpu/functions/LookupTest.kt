package shape.komputation.cpu.functions

import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Test

class LookupTest {

    private val padding = 0.0f

    @Test
    fun testOneIdOneDimension() {

        val actual = floatArrayOf(0f)
        val expected = floatArrayOf(1.0f)

        lookup(
            arrayOf(floatArrayOf(1.0f)),
            1,
            1,
            this.padding,
            intArrayOf(0),
            actual)

        assertArrayEquals(expected, actual)

    }

    @Test
    fun testOneIdTwoDimensions() {

        val actual = floatArrayOf(0f, 0f)
        val expected = floatArrayOf(1.0f, 2.0f)

        lookup(
            arrayOf(floatArrayOf(1.0f, 2.0f)),
            1,
            2,
            this.padding,
            intArrayOf(0),
            actual)

        assertArrayEquals(expected, actual)

    }

    @Test
    fun testTwoIdsTwoDimensions() {

        val actual = floatArrayOf(0f, 0f, 0f, 0f)
        val expected = floatArrayOf(1.0f, 2.0f, 3.0f, 4.0f)

        lookup(
            arrayOf(floatArrayOf(1.0f, 2.0f), floatArrayOf(3.0f, 4.0f)),
            2,
            2,
            this.padding,
            intArrayOf(0, 1),
            actual)

        assertArrayEquals(expected, actual)

    }

    @Test
    fun testTwoIdsTwoDimensionsReversed() {

        val actual = floatArrayOf(0f, 0f, 0f, 0f)
        val expected = floatArrayOf(3.0f, 4.0f, 1.0f, 2.0f)

        lookup(
            arrayOf(floatArrayOf(1.0f, 2.0f), floatArrayOf(3.0f, 4.0f)),
            2,
            2,
            this.padding,
            intArrayOf(1, 0),
            actual)

        assertArrayEquals(expected, actual)

    }

    @Test
    fun testLengthOfTwoOneIdOneDimension() {

        val actual = floatArrayOf(0f, 0f)
        val expected = floatArrayOf(1.0f, this.padding)

        lookup(
            arrayOf(floatArrayOf(1.0f, 2.0f)),
            2,
            1,
            this.padding,
            intArrayOf(0),
            actual)

        assertArrayEquals(expected, actual)

    }


}
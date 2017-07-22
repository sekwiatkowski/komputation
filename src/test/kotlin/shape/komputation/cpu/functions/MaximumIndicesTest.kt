package shape.komputation.cpu.functions

import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

class FindMaxIndicesInRowsTest {

    @Test
    fun testOneByOne() {

        val expected = intArrayOf(0)
        val actual = findMaxIndicesInRows(floatArrayOf(1.0f), 1, 1)

        assertArrayEquals(expected, actual)

    }

    @Test
    fun testTwoByOne() {

        val expected = intArrayOf(0, 1)
        val actual = findMaxIndicesInRows(floatArrayOf(1.0f, 1.0f), 2, 1)

        assertArrayEquals(expected, actual)

    }

    @Test
    fun testTwoByTwo() {

        val expected = intArrayOf(2, 3)
        val actual = findMaxIndicesInRows(floatArrayOf(1.0f, 2.0f, 3.0f, 4.0f), 2, 2)

        assertArrayEquals(expected, actual)

    }

    @Test
    fun testTwoByTwoReversed() {

        val expected = intArrayOf(0, 1)
        val actual = findMaxIndicesInRows(floatArrayOf(4.0f, 3.0f, 2.0f, 1.0f), 2, 2)

        assertArrayEquals(expected, actual)

    }


}

class FindMaxIndex {

    @Test
    fun testOneDimension() {

        assertEquals(findMaxIndex(floatArrayOf(1.0f)), 0)

    }

    @Test
    fun testTwoDimensions() {

        assertEquals(findMaxIndex(floatArrayOf(1.0f, 2.0f)), 1)
        assertEquals(findMaxIndex(floatArrayOf(2.0f, 1.0f)), 0)

    }

}
package shape.komputation.functions

import org.junit.jupiter.api.Test
import org.junit.jupiter.api.Assertions.assertArrayEquals

class MaximumIndicesTest {

    @Test
    fun testOneByOne() {

        val expected = intArrayOf(0)
        val actual = findMaxIndicesInRows(doubleArrayOf(1.0), 1, 1)

        assertArrayEquals(expected, actual)

    }

    @Test
    fun testTwoByOne() {

        val expected = intArrayOf(0, 1)
        val actual = findMaxIndicesInRows(doubleArrayOf(1.0, 1.0), 2, 1)

        assertArrayEquals(expected, actual)

    }

    @Test
    fun testTwoByTwo() {

        val expected = intArrayOf(2, 3)
        val actual = findMaxIndicesInRows(doubleArrayOf(1.0, 2.0, 3.0, 4.0), 2, 2)

        assertArrayEquals(expected, actual)

    }

    @Test
    fun testTwoByTwoReversed() {

        val expected = intArrayOf(0, 1)
        val actual = findMaxIndicesInRows(doubleArrayOf(4.0, 3.0, 2.0, 1.0), 2, 2)

        assertArrayEquals(expected, actual)

    }


}
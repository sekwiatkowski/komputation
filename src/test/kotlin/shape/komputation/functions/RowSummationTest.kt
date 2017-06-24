package shape.komputation.functions

import org.junit.jupiter.api.Test
import org.junit.jupiter.api.Assertions.assertArrayEquals

class RowSummationTest {

    @Test
    fun test1() {

        val actual = sumRows(doubleArrayOf(1.0), 1, 1)
        val expected = doubleArrayOf(1.0)

        assertArrayEquals(expected, actual)

    }

    @Test
    fun test2() {

        val actual = sumRows(doubleArrayOf(1.0, 2.0), 2, 1)
        val expected = doubleArrayOf(1.0, 2.0)

        assertArrayEquals(expected, actual)

    }

    @Test
    fun test3() {

        val actual = sumRows(doubleArrayOf(1.0, 2.0, 3.0, 4.0), 2, 2)
        val expected = doubleArrayOf(4.0, 6.0)

        assertArrayEquals(expected, actual)

    }

}
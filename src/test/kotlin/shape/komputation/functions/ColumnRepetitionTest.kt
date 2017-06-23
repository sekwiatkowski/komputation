package shape.komputation.functions

import org.junit.jupiter.api.Test
import org.junit.jupiter.api.Assertions.assertArrayEquals

class ColumnRepetitionTest {

    @Test
    fun test1() {

        val actual = repeatColumn(doubleArrayOf(1.0), 1)
        val expected = doubleArrayOf(1.0)

        assertArrayEquals(expected, actual)

    }

    @Test
    fun test2() {

        val actual = repeatColumn(doubleArrayOf(1.0), 2)
        val expected = doubleArrayOf(1.0, 1.0)

        assertArrayEquals(expected, actual)

    }

    @Test
    fun test3() {

        val actual = repeatColumn(doubleArrayOf(1.0, 2.0), 2)
        val expected = doubleArrayOf(1.0, 2.0, 1.0, 2.0)

        assertArrayEquals(expected, actual)

    }

}
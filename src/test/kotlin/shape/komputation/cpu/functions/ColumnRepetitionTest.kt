package shape.komputation.cpu.functions

import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Test

class ColumnRepetitionTest {

    @Test
    fun test1() {

        val actual = repeatColumn(floatArrayOf(1.0f), 1)
        val expected = floatArrayOf(1.0f)

        assertArrayEquals(expected, actual)

    }

    @Test
    fun test2() {

        val actual = repeatColumn(floatArrayOf(1.0f), 2)
        val expected = floatArrayOf(1.0f, 1.0f)

        assertArrayEquals(expected, actual)

    }

    @Test
    fun test3() {

        val actual = repeatColumn(floatArrayOf(1.0f, 2.0f), 2)
        val expected = floatArrayOf(1.0f, 2.0f, 1.0f, 2.0f)

        assertArrayEquals(expected, actual)

    }

}
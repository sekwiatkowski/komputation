package shape.komputation.cpu.functions

import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Test

class TranspositionTest {

    @Test
    fun test1() {

        val actual = transpose(1, 1, doubleArrayOf(1.0))
        val expected = doubleArrayOf(1.0)

        assertArrayEquals(expected, actual)

    }

    @Test
    fun test2() {

        val actual = transpose(2, 1, doubleArrayOf(1.0, 2.0))
        val expected = doubleArrayOf(1.0, 2.0)

        assertArrayEquals(expected, actual)

    }

    @Test
    fun test3() {

        val actual = transpose(1, 2, doubleArrayOf(1.0, 2.0))
        val expected = doubleArrayOf(1.0, 2.0)

        assertArrayEquals(expected, actual)

    }

    @Test
    fun test4() {

        val actual = transpose(2, 2, doubleArrayOf(1.0, 2.0, 3.0, 4.0))
        val expected = doubleArrayOf(1.0, 3.0, 2.0, 4.0)

        assertArrayEquals(expected, actual)

    }

    @Test
    fun test5() {

        val actual = transpose(3, 2, doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
        val expected = doubleArrayOf(1.0, 4.0, 2.0, 5.0, 3.0, 6.0)

        assertArrayEquals(expected, actual)

    }

    @Test
    fun test6() {

        val actual = transpose(2, 3, doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
        val expected = doubleArrayOf(1.0, 3.0, 5.0, 2.0, 4.0, 6.0)

        assertArrayEquals(expected, actual)

    }

}
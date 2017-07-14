package shape.komputation.cpu.functions.activation

import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Test

class ColumnWiseSoftmaxTest {

    @Test
    fun test1 () {

        val actual = columnWiseSoftmax(doubleArrayOf(1.0), 1, 1)
        val expected = doubleArrayOf(1.0)

        assertArrayEquals(expected, actual, 0.001)

    }

    @Test
    fun test2 () {

        val actual = columnWiseSoftmax(doubleArrayOf(1.0, 1.0), 2, 1)
        val expected = doubleArrayOf(0.5, 0.5)

        assertArrayEquals(expected, actual, 0.001)

    }

    @Test
    fun test3 () {

        val actual = columnWiseSoftmax(doubleArrayOf(1.0, 1.0, 2.0, 3.0), 2, 2)
        val denominator = Math.exp(2.0) + Math.exp(3.0)
        val expected = doubleArrayOf(0.5, 0.5, Math.exp(2.0)/ denominator, Math.exp(3.0)/ denominator)

        assertArrayEquals(expected, actual, 0.001)

    }

}
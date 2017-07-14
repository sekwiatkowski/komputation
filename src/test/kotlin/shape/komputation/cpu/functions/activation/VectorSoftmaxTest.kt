package shape.komputation.cpu.functions.activation

import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Test

class VectorSoftmaxTest {

    @Test
    fun test1 () {

        val actual = vectorSoftmax(doubleArrayOf(1.0))
        val expected = doubleArrayOf(1.0)

        assertArrayEquals(expected, actual, 0.001)

    }

    @Test
    fun test2 () {

        val actual = vectorSoftmax(doubleArrayOf(1.0, 1.0))
        val expected = doubleArrayOf(0.5, 0.5)

        assertArrayEquals(expected, actual, 0.001)

    }

    @Test
    fun test3 () {

        val actual = vectorSoftmax(doubleArrayOf(2.0, 3.0))
        val denominator = Math.exp(2.0) + Math.exp(3.0)
        val expected = doubleArrayOf(Math.exp(2.0)/ denominator, Math.exp(3.0)/ denominator)

        assertArrayEquals(expected, actual, 0.001)

    }

}
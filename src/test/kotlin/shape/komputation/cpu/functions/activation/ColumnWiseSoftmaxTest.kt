package shape.komputation.cpu.functions.activation

import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Test
import shape.komputation.matrix.FloatMath

class ColumnWiseSoftmaxTest {

    @Test
    fun test1 () {

        val actual = FloatArray(1)
        columnWiseSoftmax(floatArrayOf(1.0f), 1, 1, actual)

        val expected = floatArrayOf(1.0f)

        assertArrayEquals(expected, actual, 0.001f)

    }

    @Test
    fun test2 () {

        val actual = FloatArray(2)
        columnWiseSoftmax(floatArrayOf(1.0f, 1.0f), 2, 1, actual)

        val expected = floatArrayOf(0.5f, 0.5f)

        assertArrayEquals(expected, actual, 0.001f)

    }

    @Test
    fun test3 () {

        val actual = FloatArray(4)
        columnWiseSoftmax(floatArrayOf(1.0f, 1.0f, 2.0f, 3.0f), 2, 2, actual)
        val denominator = FloatMath.exp(2.0f) + FloatMath.exp(3.0f)
        val expected = floatArrayOf(0.5f, 0.5f, FloatMath.exp(2.0f)/ denominator, FloatMath.exp(3.0f)/ denominator)

        assertArrayEquals(expected, actual, 0.001f)

    }

}
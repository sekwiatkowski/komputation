package shape.komputation.optimization

import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Test

class StochasticGradientDescentTest {

    @Test
    fun testOneDimension() {

        val sgd = StochasticGradientDescent(0.1)

        val parameter = doubleArrayOf(1.0)

        sgd.updateDensely(parameter, doubleArrayOf(0.2), 1)

        assertArrayEquals(doubleArrayOf(0.98), parameter)

    }

    @Test
    fun testTwoDimensions() {

        val sgd = StochasticGradientDescent(0.1)

        val parameter = doubleArrayOf(1.0, 2.0)

        sgd.updateDensely(parameter, doubleArrayOf(0.1, 0.2), 2)

        assertArrayEquals(doubleArrayOf(0.99, 1.98), parameter)

    }

}
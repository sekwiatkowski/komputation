package shape.komputation.optimization.historical

import org.junit.jupiter.api.Assertions
import org.junit.jupiter.api.Test

class MomentumTest {

    @Test
    fun name() {

        val momentum = Momentum(0.1, 0.9, 1)

        val parameter = doubleArrayOf(1.0)
        val gradientSize = 1

        // history * decay - learning rate * gradient = learning rate * gradient = - 0.1 * 0.1 = -0.01
        momentum.updateDensely(parameter, doubleArrayOf(0.1), gradientSize)

        // history * decay + learning rate * gradient = 0.9 * (-0.01) - 0.1 * 0.1 = -0.009 - 0.01 = -0.019
        momentum.updateDensely(parameter, doubleArrayOf(0.1), gradientSize)

        Assertions.assertArrayEquals(doubleArrayOf(1.0 - 0.01 - 0.019), parameter)

    }
}
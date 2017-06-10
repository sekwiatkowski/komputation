package shape.konvolution.optimization

import org.junit.jupiter.api.Test
import shape.konvolution.assertMatrixEquality
import shape.konvolution.matrix.createRealMatrix

class StochasticGradientDescentTest {

    @Test
    fun test() {

        val parameter = createRealMatrix(doubleArrayOf(1.0, 2.0))

        val gradient = createRealMatrix(doubleArrayOf(1.0, 2.0))

        StochasticGradientDescent(0.1).optimize(parameter, gradient)
        val expected = createRealMatrix(doubleArrayOf(0.9, 1.8))

        assertMatrixEquality(expected, parameter, 0.01)


    }

}
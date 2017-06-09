package shape.konvolution.optimization

import org.junit.jupiter.api.Test
import shape.konvolution.assertMatrixEquality
import shape.konvolution.createDenseMatrix

class StochasticGradientDescentTest {

    @Test
    fun test() {

        val parameter = createDenseMatrix(doubleArrayOf(1.0, 2.0))

        val gradient = createDenseMatrix(doubleArrayOf(1.0, 2.0))

        StochasticGradientDescent(0.1).optimize(parameter, gradient)
        val expected = createDenseMatrix(doubleArrayOf(0.9, 1.8))

        assertMatrixEquality(expected, parameter, 0.01)


    }

}
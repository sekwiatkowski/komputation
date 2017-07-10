package shape.komputation.layers.forwarding.projection

import org.junit.jupiter.api.Test
import shape.komputation.assertMatrixEquality
import shape.komputation.layers.forward.projection.CublasProjectionLayer
import shape.komputation.matrix.DoubleMatrix
import shape.komputation.matrix.doubleScalar
import shape.komputation.optimization.DenseAccumulator

class CublasProjectionLayerTest {

    @Test
    fun testOneByOne() {

        val weights = doubleScalar(2.0)
        val input = doubleScalar(3.0)
        val expected = doubleScalar(6.0)

        check(weights, input, expected)
        check(input, weights, expected)

    }

    @Test
    fun testOneByOneWithBias() {

        val weights = doubleScalar(2.0)
        val input = doubleScalar(3.0)
        val bias = doubleArrayOf(2.0)

        val expected = doubleScalar(8.0)

        checkWithBias(weights, bias, input, expected)
        checkWithBias(input, bias, weights, expected)

    }

    @Test
    fun testOneByTwoTimesTwoByOne() {

        /*
                    3.0
                    4.0
            1.0 2.0 11.0
         */
        val weights = DoubleMatrix(1, 2, doubleArrayOf(1.0, 2.0))
        val input = DoubleMatrix(2, 1, doubleArrayOf(3.0, 4.0))

        check(weights, input, DoubleMatrix(1, 1, doubleArrayOf(11.0)))

    }

    @Test
    fun testOneByTwoTimesTwoByOneWithBias() {

        val weights = DoubleMatrix(1, 2, doubleArrayOf(1.0, 2.0))
        val bias = doubleArrayOf(5.0)
        val input = DoubleMatrix(2, 1, doubleArrayOf(3.0, 4.0))

        checkWithBias(weights, bias, input, DoubleMatrix(1, 1, doubleArrayOf(16.0)))

    }

    private fun check(weightMatrix : DoubleMatrix, inputMatrix: DoubleMatrix, expected : DoubleMatrix) {

        val weightAccumulator = DenseAccumulator(weightMatrix.numberRows * weightMatrix.numberColumns)

        val layer = CublasProjectionLayer(null, weightMatrix.entries, weightMatrix.numberRows, weightMatrix.numberColumns, weightAccumulator)
        val actual = layer.forward(inputMatrix, false)

        assertMatrixEquality(expected, actual, 0.001)

    }

    private fun checkWithBias(weightMatrix : DoubleMatrix, bias : DoubleArray, inputMatrix: DoubleMatrix, expected : DoubleMatrix) {

        val weightAccumulator = DenseAccumulator(weightMatrix.numberRows * weightMatrix.numberColumns)

        val layer = CublasProjectionLayer(null, weightMatrix.entries, weightMatrix.numberRows, weightMatrix.numberColumns, weightAccumulator, null, bias)
        val actual = layer.forward(inputMatrix, false)

        assertMatrixEquality(expected, actual, 0.001)

    }

}
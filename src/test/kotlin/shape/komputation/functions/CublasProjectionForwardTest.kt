package shape.komputation.functions

import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Test
import shape.komputation.matrix.DoubleMatrix
import shape.komputation.matrix.doubleScalar

class CublasProjectionForwardTest {

    @Test
    fun testOneByOne() {

        val weights = doubleScalar(2.0)
        val input = doubleScalar(3.0)
        val expected = doubleArrayOf(6.0)

        check(weights, input, expected)
        check(input, weights, expected)

    }

    @Test
    fun testOneByOneWithBias() {

        val weights = doubleScalar(2.0)
        val input = doubleScalar(3.0)
        val bias = doubleArrayOf(2.0)

        val expected = doubleArrayOf(8.0)

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

        check(weights, input, doubleArrayOf(11.0))

    }

    @Test
    fun testOneByTwoTimesTwoByOneWithBias() {

        val weights = DoubleMatrix(1, 2, doubleArrayOf(1.0, 2.0))
        val bias = doubleArrayOf(5.0)
        val input = DoubleMatrix(2, 1, doubleArrayOf(3.0, 4.0))

        checkWithBias(weights, bias, input, doubleArrayOf(16.0))

    }

    private fun check(weightMatrix : DoubleMatrix, inputMatrix: DoubleMatrix, expected : DoubleArray) {

        val actual = cublasProject(
            inputMatrix.entries,
            weightMatrix.numberRows,
            weightMatrix.numberColumns,
            weightMatrix.numberRows * weightMatrix.numberColumns,
            weightMatrix.entries)

        assertArrayEquals(expected, actual, 0.001)

    }

    private fun checkWithBias(weightMatrix : DoubleMatrix, bias : DoubleArray, inputMatrix: DoubleMatrix, expected : DoubleArray) {

        val actual = cublasProject(
            inputMatrix.entries,
            weightMatrix.numberRows,
            weightMatrix.numberColumns,
            weightMatrix.numberRows * weightMatrix.numberColumns,
            weightMatrix.entries,
            bias)

        assertArrayEquals(expected, actual, 0.001)

    }

}
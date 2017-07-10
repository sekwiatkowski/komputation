package shape.komputation.functions

import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Test
import shape.komputation.matrix.DoubleMatrix
import shape.komputation.matrix.doubleColumnVector
import shape.komputation.matrix.doubleScalar

class CublasProjectionLayerTest {

    @Test
    fun testBackwardProjectionWrtInput1() {

        val weights = doubleScalar(2.0)
        val chain = doubleArrayOf(3.0)
        val expected = doubleArrayOf(6.0)

        checkBackwardProjectionWrtInput(weights, chain, expected)

    }

    @Test
    fun testBackwardProjectionWrtInput2() {

        /*
            weights = 2
                      3
            chain = 4
                    5

                             4
                             5
            weights^T >> 2 3
         */

        val weights = doubleColumnVector(2.0, 3.0)
        val chain = doubleArrayOf(4.0, 5.0)
        val expected = doubleArrayOf(2.0*4.0 + 3.0*5.0)

        checkBackwardProjectionWrtInput(weights, chain, expected)

    }

    @Test
    fun testBackwardProjectionWrtInput3() {

        /*
            weights = 2 3
                      4 5
            chain = 6
                    7

                              6
                              7
            weights^T >> 2 4 40
                         3 5 53
         */

        val weights = DoubleMatrix(2, 2, doubleArrayOf(2.0, 4.0, 3.0, 5.0))
        val chain = doubleArrayOf(6.0, 7.0)
        val expected = doubleArrayOf(40.0, 53.0)

        checkBackwardProjectionWrtInput(weights, chain, expected)

    }

    private fun checkBackwardProjectionWrtInput(weights: DoubleMatrix, chain: DoubleArray, expected: DoubleArray) {

        val actual = cublasBackwardProjectionWrtInput(weights.entries, weights.numberRows, weights.numberColumns, weights.numberRows * weights.numberColumns, chain)
        assertArrayEquals(expected, actual, 0.001)

    }

}
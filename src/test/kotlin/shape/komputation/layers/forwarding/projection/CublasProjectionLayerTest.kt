package shape.komputation.layers.forwarding.projection

import org.junit.jupiter.api.Test
import shape.komputation.assertMatrixEquality
import shape.komputation.layers.forward.projection.CublasProjectionLayer
import shape.komputation.matrix.DoubleMatrix
import shape.komputation.matrix.doubleScalar

class CublasProjectionLayerTest {

    @Test
    fun testOneByOne() {

        val A = doubleScalar(2.0)
        val B = doubleScalar(3.0)
        val expected = doubleScalar(6.0)

        check(A, B, expected)
        check(B, A, expected)

    }

    @Test
    fun testTwoByTwo() {

        val A = DoubleMatrix(2, 2, doubleArrayOf(1.0, 3.0, 2.0, 4.0))
        val B = DoubleMatrix(2, 2, doubleArrayOf(5.0, 7.0, 6.0, 8.0))

        check(A, B, DoubleMatrix(2, 2, doubleArrayOf(19.0, 43.0, 22.0, 50.0)))
        check(B, A, DoubleMatrix(2, 2, doubleArrayOf(23.0, 31.0, 34.0, 46.0)))

    }

    @Test
    fun testOneByTwo_TwoByOne() {

        /*
                    3.0
                    4.0
            1.0 2.0 11.0
         */
        val A = DoubleMatrix(1, 2, doubleArrayOf(1.0, 2.0))
        val B = DoubleMatrix(2, 1, doubleArrayOf(3.0, 4.0))

        check(A, B, DoubleMatrix(1, 1, doubleArrayOf(11.0)))

    }

    @Test
    fun testTwoByOne_OneByTwo() {

        /*
                3.0 4.0
            1.0 3.0 4.0
            2.0 6.0 8.0
        */
        val A = DoubleMatrix(2, 1, doubleArrayOf(1.0, 2.0))
        val B = DoubleMatrix(1, 2, doubleArrayOf(3.0, 4.0))

        check(A, B, DoubleMatrix(2, 2, doubleArrayOf(3.0, 6.0, 4.0, 8.0)))

    }

    private fun check(weightMatrix : DoubleMatrix, inputMatrix: DoubleMatrix, expected : DoubleMatrix) {

        val layer = CublasProjectionLayer(null, weightMatrix.entries, weightMatrix.numberRows, weightMatrix.numberColumns)
        val actual = layer.forward(inputMatrix, false)

        assertMatrixEquality(expected, actual, 0.001)
    }

}
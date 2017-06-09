package shape.konvolution.layers

import org.junit.jupiter.api.Test
import shape.konvolution.assertMatrixEquality
import shape.konvolution.createDenseMatrix

class MaxPoolingLayerTest {

    @Test
    fun test() {

        val maxPoolingLayer = MaxPoolingLayer()

        val input = createDenseMatrix(
            doubleArrayOf(1.0, 2.0),
            doubleArrayOf(3.0, -4.0)
        )

        val actual = maxPoolingLayer.forward(input)

        val expected = createDenseMatrix(
            doubleArrayOf(2.0),
            doubleArrayOf(3.0)
        )

        assertMatrixEquality(expected, actual, 0.001)

    }



}
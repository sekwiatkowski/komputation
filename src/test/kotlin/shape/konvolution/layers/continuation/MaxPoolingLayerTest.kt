package shape.konvolution.layers.continuation

import org.junit.jupiter.api.Test
import shape.konvolution.assertMatrixEquality
import shape.konvolution.matrix.createRealMatrix

class MaxPoolingLayerTest {

    @Test
    fun test() {

        val maxPoolingLayer = MaxPoolingLayer()

        val input = createRealMatrix(
            doubleArrayOf(1.0, 2.0),
            doubleArrayOf(3.0, -4.0)
        )

        maxPoolingLayer.setInput(input)
        maxPoolingLayer.forward()

        val expected = createRealMatrix(
            doubleArrayOf(2.0),
            doubleArrayOf(3.0)
        )

        val actual = maxPoolingLayer.lastForwardResult.last()

        assertMatrixEquality(expected, actual, 0.001)

    }

}
package shape.komputation.layers.continuation

import org.junit.jupiter.api.Test
import shape.komputation.assertMatrixEquality
import shape.komputation.layers.continuation.activation.SoftmaxLayer
import shape.komputation.matrix.createRealMatrix

class SoftmaxLayerTest {

    @Test
    fun test() {

        val softmaxLayer = SoftmaxLayer()

        val input = createRealMatrix(doubleArrayOf(1.0, 2.0), doubleArrayOf(3.0, 4.0))

        softmaxLayer.setInput(input)
        softmaxLayer.forward()

        val expected = createRealMatrix(
            doubleArrayOf(Math.exp(1.0) / (Math.exp(1.0) + Math.exp(3.0)), Math.exp(2.0) / (Math.exp(2.0) + Math.exp(4.0))),
            doubleArrayOf(Math.exp(3.0) / (Math.exp(1.0) + Math.exp(3.0)), Math.exp(4.0) / (Math.exp(2.0) + Math.exp(4.0)))
        )

        val actual = softmaxLayer.lastForwardResult.last()

        assertMatrixEquality(expected, actual, 0.0001)

    }


}
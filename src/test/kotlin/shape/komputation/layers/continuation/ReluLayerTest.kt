package shape.komputation.layers.continuation

import org.junit.jupiter.api.Test
import shape.komputation.assertMatrixEquality
import shape.komputation.matrix.createRealMatrix

class ReluLayerTest {

    @Test
    fun testForward() {

        val reluLayer = ReluLayer()

        val input = createRealMatrix(doubleArrayOf(1.0, -2.0), doubleArrayOf(-3.0, 4.0))

        reluLayer.setInput(input)
        reluLayer.forward()

        val expected = createRealMatrix(
            doubleArrayOf(1.0, 0.0),
            doubleArrayOf(0.0, 4.0)
        )

        assertMatrixEquality(expected, reluLayer.lastForwardResult.last(), 0.0001)

    }

    @Test
    fun testBackward() {

        val reluLayer = ReluLayer()

        val input = createRealMatrix(doubleArrayOf(-1.0), doubleArrayOf(2.0))

        reluLayer.setInput(input)
        reluLayer.forward()

        val expected = createRealMatrix(
            doubleArrayOf(0.0),
            doubleArrayOf(2.0)
        )

        val actual = reluLayer.lastForwardResult.last()

        assertMatrixEquality(expected, actual, 0.0001)

    }


}
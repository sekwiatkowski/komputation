package shape.komputation.layers.continuation

import org.junit.jupiter.api.Test
import shape.komputation.assertMatrixEquality
import shape.komputation.matrix.createRealMatrix

class ProjectionLayerTest {

    @Test
    fun testForward() {

        val projectionLayer = ProjectionLayer(null, createRealMatrix(doubleArrayOf(1.0, 2.0, 3.0), doubleArrayOf(4.0, 5.0, 6.0)))

        val input = createRealMatrix(doubleArrayOf(1.0), doubleArrayOf(2.0), doubleArrayOf(3.0))

        projectionLayer.setInput(input)
        projectionLayer.forward()

        val expected = createRealMatrix(
            doubleArrayOf(1.0 * 1.0 + 2.0 * 2.0 + 3.0 * 3.0),
            doubleArrayOf(4.0 * 1.0 + 5.0 * 2.0 + 6.0 * 3.0)
        )

        val actual = projectionLayer.lastForwardResult.last()

        assertMatrixEquality(expected, actual, 0.001)

    }

}
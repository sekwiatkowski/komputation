package shape.konvolution.layers.continuation

import org.junit.jupiter.api.Test
import shape.konvolution.assertMatrixEquality
import shape.konvolution.matrix.createRealMatrix

class ProjectionLayerTest {

    @Test
    fun testForward() {

        val projectionLayer = ProjectionLayer(createRealMatrix(doubleArrayOf(1.0, 2.0, 3.0), doubleArrayOf(4.0, 5.0, 6.0)))

        val input = createRealMatrix(doubleArrayOf(1.0), doubleArrayOf(2.0), doubleArrayOf(3.0))

        val actual = projectionLayer.forward(input)
        val expected = createRealMatrix(
            doubleArrayOf(1.0 * 1.0 + 2.0 * 2.0 + 3.0 * 3.0),
            doubleArrayOf(4.0 * 1.0 + 5.0 * 2.0 + 6.0 * 3.0)
        )

        assertMatrixEquality(expected, actual, 0.001)

    }

}
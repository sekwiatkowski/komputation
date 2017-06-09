package shape.konvolution.layers

import org.junit.jupiter.api.Test
import shape.konvolution.assertMatrixEquality
import shape.konvolution.createDenseMatrix

class ProjectionLayerTest {

    @Test
    fun testForward() {

        val projectionLayer = ProjectionLayer(createDenseMatrix(doubleArrayOf(1.0, 2.0, 3.0), doubleArrayOf(4.0, 5.0, 6.0)))

        val input = createDenseMatrix(doubleArrayOf(1.0), doubleArrayOf(2.0), doubleArrayOf(3.0))

        val actual = projectionLayer.forward(input)
        val expected = createDenseMatrix(
            doubleArrayOf(1.0*1.0+2.0*2.0+3.0*3.0),
            doubleArrayOf(4.0*1.0+5.0*2.0+6.0*3.0)
        )

        assertMatrixEquality(expected, actual, 0.001)

    }

}
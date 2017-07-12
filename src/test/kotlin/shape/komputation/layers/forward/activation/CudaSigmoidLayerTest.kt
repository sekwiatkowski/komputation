package shape.komputation.layers.forward.activation

import org.junit.jupiter.api.Test
import shape.komputation.assertMatrixEquality
import shape.komputation.cuda.initializeCuda
import shape.komputation.matrix.doubleColumnVector
import shape.komputation.matrix.doubleRowVector
import shape.komputation.matrix.doubleScalar

class CudaSigmoidLayerTest {

    @Test
    fun testOneDimension() {

        initializeCuda()

        val layer = cudaSigmoidLayer(1, 1, 1)
        layer.acquire()

        val actual = layer.forward(doubleScalar(0.0), false)
        val expected = doubleScalar(0.5)

        layer.release()

        assertMatrixEquality(expected, actual, 0.001)

    }

    @Test
    fun testTwoDimensions() {

        initializeCuda()

        val layer = cudaSigmoidLayer(2, 2, 1)
        layer.acquire()

        val actual = layer.forward(doubleRowVector(0.0, 1.0), false)
        val expected = doubleColumnVector(0.5, 0.731058579)

        layer.release()

        assertMatrixEquality(expected, actual, 0.001)

    }


}
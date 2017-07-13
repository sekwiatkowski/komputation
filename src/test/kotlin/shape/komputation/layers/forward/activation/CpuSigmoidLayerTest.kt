package shape.komputation.layers.forward.activation

import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Test
import shape.komputation.functions.activation.sigmoid
import shape.komputation.matrix.doubleColumnVector

class CpuSigmoidLayerTest {

    private val layer = CpuSigmoidLayer()

    @Test
    fun testSparseForwarding() {

        val actual = layer
            .forward(doubleColumnVector(0.25, 0.5, 0.75), booleanArrayOf(false, false, true))
            .entries

        val expected = doubleArrayOf(0.0, 0.0, sigmoid(0.75))

        assertArrayEquals(actual, expected)

    }

}
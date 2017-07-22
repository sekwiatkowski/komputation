package shape.komputation.cpu.layers.forward.activation

import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Test
import shape.komputation.cpu.functions.activation.sigmoid
import shape.komputation.matrix.floatColumnVector

class CpuSigmoidLayerTest {

    private val layer = CpuSigmoidLayer()

    @Test
    fun testSparseForwarding() {

        val actual = layer
            .forward(floatColumnVector(0.25f, 0.5f, 0.75f), booleanArrayOf(false, false, true))
            .entries

        val expected = floatArrayOf(0.0f, 0.0f, sigmoid(0.75f))

        assertArrayEquals(actual, expected)

    }

}
package shape.komputation.cpu.layers.forward.activation

import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Test
import shape.komputation.matrix.floatColumnVector

class CpuReluLayerTest {

    private val layer = CpuReluLayer()

    @Test
    fun testSparseForwarding() {

        val actual = layer
            .forward(floatColumnVector(-1.0f, 1.0f, -2.0f, 2.0f), booleanArrayOf(false, true, true, false))
            .entries

        val expected = floatArrayOf(0.0f, 1.0f, 0.0f, 0.0f)

        assertArrayEquals(actual, expected)

    }

}
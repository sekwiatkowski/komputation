package shape.komputation.cpu.layers.forward.activation

import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Test
import shape.komputation.matrix.FloatMath
import shape.komputation.matrix.floatColumnVector

class CpuTanhLayerTest {

    private val layer = CpuTanhLayer()

    @Test
    fun testSparseForwarding() {

        val actual = layer
            .forward(floatColumnVector(-0.5f, 0.0f, 0.5f), booleanArrayOf(false, false, true))
            .entries

        val expected = floatArrayOf(0.0f, 0.0f, FloatMath.tanh(0.5f))

        assertArrayEquals(actual, expected)

    }

}
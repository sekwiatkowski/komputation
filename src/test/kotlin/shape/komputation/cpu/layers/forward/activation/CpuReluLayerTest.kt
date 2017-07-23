package shape.komputation.cpu.layers.forward.activation

import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Test
import shape.komputation.cpu.functions.activation.relu
import shape.komputation.layers.forward.activation.reluLayer
import shape.komputation.matrix.floatColumnVector

class CpuReluLayerTest {

    @Test
    fun testSparseForwarding() {

        val layer = reluLayer(4).buildForCpu()

        val actual = layer
            .forward(floatColumnVector(-1.0f, 1.0f, -2.0f, 2.0f), booleanArrayOf(false, true, true, false))
            .entries

        val expected = floatArrayOf(0.0f, 1.0f, 0.0f, 0.0f)

        assertArrayEquals(actual, expected)

    }

}
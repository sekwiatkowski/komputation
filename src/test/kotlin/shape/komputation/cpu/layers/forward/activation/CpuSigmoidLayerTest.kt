package shape.komputation.cpu.layers.forward.activation

import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Test
import shape.komputation.cpu.functions.activation.sigmoid
import shape.komputation.layers.forward.activation.sigmoidLayer
import shape.komputation.matrix.floatColumnVector

class CpuSigmoidLayerTest {

    @Test
    fun testSparseForwarding() {

        val layer = sigmoidLayer(3).buildForCpu()

        val actual = layer
            .forward(floatColumnVector(0.25f, 0.5f, 0.75f), booleanArrayOf(false, false, true))
            .entries

        val expected = floatArrayOf(0.0f, 0.0f, sigmoid(0.75f))

        assertArrayEquals(actual, expected)

    }

}
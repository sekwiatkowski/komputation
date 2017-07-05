package shape.komputation.layers.forwarding.activation

import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Test
import shape.komputation.functions.activation.tanh
import shape.komputation.layers.forward.activation.tanhLayer
import shape.komputation.matrix.doubleColumnVector

class TanhLayerTest {

    private val layer = tanhLayer()

    @Test
    fun testSparseForwarding() {

        val actual = layer
            .forward(doubleColumnVector(-0.5, 0.0, 0.5), booleanArrayOf(false, false, true))
            .entries

        val expected = doubleArrayOf(0.0, 0.0, tanh(0.5))

        assertArrayEquals(actual, expected)

    }

}
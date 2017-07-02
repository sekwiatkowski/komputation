package shape.komputation.layers.forwarding.activation

import org.junit.jupiter.api.Test
import org.junit.jupiter.api.Assertions.assertArrayEquals
import shape.komputation.layers.forward.activation.ReluLayer
import shape.komputation.matrix.doubleColumnVector

class ReluLayerTest {

    private val layer = ReluLayer()

    @Test
    fun testSparseForwarding() {

        val actual = layer
            .forward(doubleColumnVector(-1.0, 1.0, -2.0, 2.0), booleanArrayOf(false, true, true, false))
            .entries

        val expected = doubleArrayOf(0.0, 1.0, 0.0, 0.0)

        assertArrayEquals(actual, expected)

    }

}
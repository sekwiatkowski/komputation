package shape.komputation.layers.forwarding.activation

import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Test
import shape.komputation.layers.forward.activation.reluLayer
import shape.komputation.matrix.doubleColumnVector

class ReluLayerTest {

    private val layer = reluLayer()

    @Test
    fun testSparseForwarding() {

        val actual = layer
            .forward(doubleColumnVector(-1.0, 1.0, -2.0, 2.0), booleanArrayOf(false, true, true, false))
            .entries

        val expected = doubleArrayOf(0.0, 1.0, 0.0, 0.0)

        assertArrayEquals(actual, expected)

    }

}
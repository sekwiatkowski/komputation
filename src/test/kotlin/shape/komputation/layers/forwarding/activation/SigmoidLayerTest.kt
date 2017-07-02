package shape.komputation.layers.forwarding.activation

import org.junit.jupiter.api.Test
import org.junit.jupiter.api.Assertions.assertArrayEquals
import shape.komputation.functions.activation.sigmoid
import shape.komputation.layers.forward.activation.SigmoidLayer
import shape.komputation.matrix.doubleColumnVector

class SigmoidLayerTest {

    private val layer = SigmoidLayer()

    @Test
    fun testSparseForwarding() {

        val actual = layer
            .forward(doubleColumnVector(0.25, 0.5, 0.75), booleanArrayOf(false, false, true))
            .entries

        val expected = doubleArrayOf(0.0, 0.0, sigmoid(0.75))

        assertArrayEquals(actual, expected)

    }

}
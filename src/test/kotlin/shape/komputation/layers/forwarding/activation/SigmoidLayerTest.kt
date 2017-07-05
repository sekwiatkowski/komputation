package shape.komputation.layers.forwarding.activation

import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Test
import shape.komputation.functions.activation.sigmoid
import shape.komputation.layers.forward.activation.sigmoidLayer
import shape.komputation.matrix.doubleColumnVector

class SigmoidLayerTest {

    private val layer = sigmoidLayer()

    @Test
    fun testSparseForwarding() {

        val actual = layer
            .forward(doubleColumnVector(0.25, 0.5, 0.75), booleanArrayOf(false, false, true))
            .entries

        val expected = doubleArrayOf(0.0, 0.0, sigmoid(0.75))

        assertArrayEquals(actual, expected)

    }

}
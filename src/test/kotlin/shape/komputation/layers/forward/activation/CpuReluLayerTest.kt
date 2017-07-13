package shape.komputation.layers.forward.activation

import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Test
import shape.komputation.matrix.doubleColumnVector

class CpuReluLayerTest {

    private val layer = CpuReluLayer()

    @Test
    fun testSparseForwarding() {

        val actual = layer
            .forward(doubleColumnVector(-1.0, 1.0, -2.0, 2.0), booleanArrayOf(false, true, true, false))
            .entries

        val expected = doubleArrayOf(0.0, 1.0, 0.0, 0.0)

        assertArrayEquals(actual, expected)

    }

}
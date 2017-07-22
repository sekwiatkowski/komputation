package shape.komputation.cpu.layers.entry

import org.junit.jupiter.api.Test
import shape.komputation.assertMatrixEquality
import shape.komputation.layers.entry.lookupLayer
import shape.komputation.matrix.*

class CpuLookupLayerTest {

    @Test
    fun testOneDimension() {

        val firstVector = floatArrayOf(1.0f)

        val lookupLayer = lookupLayer(arrayOf(firstVector), 1, 1, 1).buildForCpu()

        val expected = floatScalar(1.0f)
        val actual = lookupLayer.forward(intScalar(0))

        assertMatrixEquality(expected, actual, 0.001f)

    }

    @Test
    fun testTwoDimensions() {

        val lookupLayer = lookupLayer(arrayOf(floatArrayOf(1.0f, 2.0f)), 2, 1, 1).buildForCpu()

        val expected = floatColumnVector(1.0f, 2.0f)
        val actual = lookupLayer.forward(intScalar(0))

        assertMatrixEquality(expected, actual, 0.001f)

    }

    @Test
    fun testOneOutOfTwoVectors() {

        val lookupLayer = lookupLayer(arrayOf(floatArrayOf(1.0f), floatArrayOf(2.0f)), 1, 1, 1).buildForCpu()

        assertMatrixEquality(
            floatScalar(1.0f),
            lookupLayer.forward(intScalar(0)),
            0.001f)

        assertMatrixEquality(
            floatScalar(2.0f),
            lookupLayer.forward(intScalar(1)),
            0.001f)

    }

    @Test
    fun testTwoOutOfTwoVectors() {

        val lookupLayer = lookupLayer(arrayOf(floatArrayOf(1.0f), floatArrayOf(2.0f)), 1, 1, 2).buildForCpu()

        assertMatrixEquality(
            floatRowVector(1.0f, 2.0f),
            lookupLayer.forward(intColumnVector(0, 1)),
            0.001f)

    }

    @Test
    fun testTwoOutOfTwoVectorsTwoDimensions() {

        val lookupLayer = lookupLayer(arrayOf(floatArrayOf(1.0f, 2.0f), floatArrayOf(3.0f, 4.0f)), 2, 1, 2).buildForCpu()

        assertMatrixEquality(
            floatMatrixFromColumns(
                floatArrayOf(1.0f, 2.0f),
                floatArrayOf(3.0f, 4.0f)
            ),
            lookupLayer.forward(intColumnVector(0, 1)),
            0.001f)

    }

    @Test
    fun testTwoOutOfTwoVectorsTwoDimensionsReversed() {

        val lookupLayer = lookupLayer(arrayOf(floatArrayOf(1.0f, 2.0f), floatArrayOf(3.0f, 4.0f)), 2, 1, 2).buildForCpu()

        assertMatrixEquality(
            floatMatrixFromColumns(
                floatArrayOf(3.0f, 4.0f),
                floatArrayOf(1.0f, 2.0f)
            ),
            lookupLayer.forward(intColumnVector(1, 0)),
            0.001f)

    }

}
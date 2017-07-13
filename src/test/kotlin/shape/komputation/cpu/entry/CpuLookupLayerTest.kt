package shape.komputation.cpu.entry

import org.junit.jupiter.api.Test
import shape.komputation.assertMatrixEquality
import shape.komputation.layers.entry.lookupLayer
import shape.komputation.matrix.*

class CpuLookupLayerTest {

    @Test
    fun testOneDimension() {

        val firstVector = doubleArrayOf(1.0)

        val lookupLayer = lookupLayer(arrayOf(firstVector), 1, 1, 1).buildForCpu()

        val expected = doubleScalar(1.0)
        val actual = lookupLayer.forward(intScalar(0))

        assertMatrixEquality(expected, actual, 0.001)

    }

    @Test
    fun testTwoDimensions() {

        val lookupLayer = lookupLayer(arrayOf(doubleArrayOf(1.0, 2.0)), 2, 1, 1).buildForCpu()

        val expected = doubleColumnVector(1.0, 2.0)
        val actual = lookupLayer.forward(intScalar(0))

        assertMatrixEquality(expected, actual, 0.001)

    }

    @Test
    fun testOneOutOfTwoVectors() {

        val lookupLayer = lookupLayer(arrayOf(doubleArrayOf(1.0), doubleArrayOf(2.0)), 1, 1, 1).buildForCpu()

        assertMatrixEquality(
            doubleScalar(1.0),
            lookupLayer.forward(intScalar(0)),
            0.001)

        assertMatrixEquality(
            doubleScalar(2.0),
            lookupLayer.forward(intScalar(1)),
            0.001)

    }

    @Test
    fun testTwoOutOfTwoVectors() {

        val lookupLayer = lookupLayer(arrayOf(doubleArrayOf(1.0), doubleArrayOf(2.0)), 1, 1, 2).buildForCpu()

        assertMatrixEquality(
            doubleRowVector(1.0, 2.0),
            lookupLayer.forward(intVector(0, 1)),
            0.001)

    }

    @Test
    fun testTwoOutOfTwoVectorsTwoDimensions() {

        val lookupLayer = lookupLayer(arrayOf(doubleArrayOf(1.0, 2.0), doubleArrayOf(3.0, 4.0)), 2, 1, 2).buildForCpu()

        assertMatrixEquality(
            doubleMatrixFromColumns(
                doubleArrayOf(1.0, 2.0),
                doubleArrayOf(3.0, 4.0)
            ),
            lookupLayer.forward(intVector(0, 1)),
            0.001)

    }

    @Test
    fun testTwoOutOfTwoVectorsTwoDimensionsReversed() {

        val lookupLayer = lookupLayer(arrayOf(doubleArrayOf(1.0, 2.0), doubleArrayOf(3.0, 4.0)), 2, 1, 2).buildForCpu()

        assertMatrixEquality(
            doubleMatrixFromColumns(
                doubleArrayOf(3.0, 4.0),
                doubleArrayOf(1.0, 2.0)
            ),
            lookupLayer.forward(intVector(1, 0)),
            0.001)

    }

}
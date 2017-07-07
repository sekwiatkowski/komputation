package shape.komputation.layers.entry

import org.junit.jupiter.api.Test
import shape.komputation.assertMatrixEquality
import shape.komputation.matrix.*

class LookupLayerTest {

    @Test
    fun testOneDimension() {

        val firstVector = doubleArrayOf(1.0)

        val lookupLayer = lookupLayer(arrayOf(firstVector), 1, 1, 1)

        val expected = doubleScalar(1.0)
        val actual = lookupLayer.forward(intScalar(0))

        assertMatrixEquality(expected, actual, 0.001)

    }

    @Test
    fun testTwoDimensions() {

        val lookupLayer = lookupLayer(arrayOf(doubleArrayOf(1.0, 2.0)), 2, 1, 1)

        val expected = doubleColumnVector(1.0, 2.0)
        val actual = lookupLayer.forward(intScalar(0))

        assertMatrixEquality(expected, actual, 0.001)

    }

    @Test
    fun testOneOutOfTwoVectors() {

        val lookupLayer = lookupLayer(arrayOf(doubleArrayOf(1.0), doubleArrayOf(2.0)), 1, 1, 1)

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

        val lookupLayer = lookupLayer(arrayOf(doubleArrayOf(1.0), doubleArrayOf(2.0)), 1, 1, 2)

        assertMatrixEquality(
            doubleRowVector(1.0, 2.0),
            lookupLayer.forward(intVector(0, 1)),
            0.001)

    }

    @Test
    fun testTwoOutOfTwoVectorsTwoDimensions() {

        val lookupLayer = lookupLayer(arrayOf(doubleArrayOf(1.0, 2.0), doubleArrayOf(3.0, 4.0)), 2, 1, 2)

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

        val lookupLayer = lookupLayer(arrayOf(doubleArrayOf(1.0, 2.0), doubleArrayOf(3.0, 4.0)), 2, 1, 2)

        assertMatrixEquality(
            doubleMatrixFromColumns(
                doubleArrayOf(3.0, 4.0),
                doubleArrayOf(1.0, 2.0)
            ),
            lookupLayer.forward(intVector(1, 0)),
            0.001)

    }

}
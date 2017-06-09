package shape.konvolution.layers

import org.junit.jupiter.api.Test
import org.junit.jupiter.api.Assertions.assertEquals
import shape.konvolution.assertMatrixEquality
import shape.konvolution.createRealMatrix
import shape.konvolution.layers.continuation.ConvolutionLayer
import shape.konvolution.layers.continuation.expandMatrixForConvolution
import shape.konvolution.layers.continuation.expandedColumnToOriginalColumn
import shape.konvolution.layers.continuation.expandedRowToOriginalRow

class ConvolutionLayerTest {

    @Test
    fun testConvolution() {

        /*
                1 2
                2 3
            1 2 5 8
            3 4 11 18
         */

        val input = createRealMatrix(
            doubleArrayOf(1.0),
            doubleArrayOf(2.0),
            doubleArrayOf(3.0))

        val weights = createRealMatrix(doubleArrayOf(1.0, 2.0), doubleArrayOf(3.0, 4.0))
        val convolutionLayer = ConvolutionLayer(1, 2, weights)

        val expected = createRealMatrix(doubleArrayOf(5.0, 8.0), doubleArrayOf(11.0, 18.0))
        val actual = convolutionLayer.forward(input)

        assertMatrixEquality(expected, actual, 0.001)

    }

    @Test
    fun testExpansion_1x1_1W1H() {

        val input = createRealMatrix(doubleArrayOf(1.0))

        val expected = createRealMatrix(doubleArrayOf(1.0))

        val actual = expandMatrixForConvolution(input, 1, 1)

        assertMatrixEquality(
            expected,
            actual,
            0.001
        )

    }

    @Test
    fun testExpansion_2x1_1W1H() {

        val input = createRealMatrix(
            doubleArrayOf(1.0),
            doubleArrayOf(2.0)
        )

        val expected = createRealMatrix(
            doubleArrayOf(1.0, 2.0)
        )

        val actual = expandMatrixForConvolution(input, 1, 1)

        assertMatrixEquality(
            expected,
            actual,
            0.001
        )

    }

    @Test
    fun testExpansion3_2x1_1W2H() {

        val input = createRealMatrix(
            doubleArrayOf(1.0),
            doubleArrayOf(2.0)
        )

        val expected = createRealMatrix(
            doubleArrayOf(1.0),
            doubleArrayOf(2.0)
        )

        val actual = expandMatrixForConvolution(input, 1, 2)

        assertMatrixEquality(
            expected,
            actual,
            0.001
        )


    }

    @Test
    fun testExpansion3_1x2_1W1H() {

        val input = createRealMatrix(
            doubleArrayOf(1.0, 2.0)
        )

        val expected = createRealMatrix(
            doubleArrayOf(1.0, 2.0)
        )

        val actual = expandMatrixForConvolution(input, 1, 1)

        assertMatrixEquality(
            expected,
            actual,
            0.001
        )


    }

    @Test
    fun testExpansion3_1x2_2W1H() {

        val input = createRealMatrix(
            doubleArrayOf(1.0, 2.0)
        )

        val expected = createRealMatrix(
            doubleArrayOf(1.0),
            doubleArrayOf(2.0)
        )

        val actual = expandMatrixForConvolution(input, 2, 1)

        assertMatrixEquality(
            expected,
            actual,
            0.001
        )


    }

    @Test
    fun testExpansion3_3x1_1W2H() {

        val input = createRealMatrix(
            doubleArrayOf(1.0),
            doubleArrayOf(2.0),
            doubleArrayOf(3.0)
        )

        val expected = createRealMatrix(
            doubleArrayOf(1.0, 2.0),
            doubleArrayOf(2.0, 3.0)
        )

        val actual = expandMatrixForConvolution(input, 1, 2)

        assertMatrixEquality(
            expected,
            actual,
            0.001
        )


    }

    @Test
    fun testExpandedRowToOriginalRow() {

        assertEquals(
            0,
            expandedRowToOriginalRow(0, 0, 1, 1)
        )

        assertEquals(
            1,
            expandedRowToOriginalRow(1, 0, 1, 1)
        )

        /*
            1 2
            3 4
         */

        assertEquals(
            0,
            expandedRowToOriginalRow(0, 0, 1, 2)
        )

        assertEquals(
            0,
            expandedRowToOriginalRow(0, 1, 1, 2)
        )

        assertEquals(
            1,
            expandedRowToOriginalRow(1, 0, 1, 2)
        )

    }

    @Test
    fun testExpandedColumnToOriginalColumn() {

        assertEquals(
            0,
            expandedColumnToOriginalColumn(0, 0, 1, 1)
        )

        assertEquals(
            0,
            expandedColumnToOriginalColumn(0, 0, 1, 2)
        )

        assertEquals(
            1,
            expandedColumnToOriginalColumn(0, 1, 1, 2)
        )

        assertEquals(
            0,
            expandedColumnToOriginalColumn(1, 0, 1, 2)
        )

        assertEquals(
            1,
            expandedColumnToOriginalColumn(1, 1, 1, 2)
        )

    }

}
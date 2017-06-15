package shape.komputation.layers.feedforward

import org.junit.jupiter.api.Test
import org.junit.jupiter.api.Assertions.assertEquals
import shape.komputation.assertMatrixEquality
import shape.komputation.layers.feedforward.convolution.expandMatrixForConvolution
import shape.komputation.layers.feedforward.convolution.expandedColumnToOriginalColumn
import shape.komputation.layers.feedforward.convolution.expandedRowToOriginalRow
import shape.komputation.matrix.createRealMatrix

class ConvolutionLayerTest {

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
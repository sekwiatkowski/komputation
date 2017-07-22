package shape.komputation.cpu.functions

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test
import shape.komputation.assertMatrixEquality
import shape.komputation.matrix.floatMatrixFromColumns

class ConvolutionExpansionTest {

    @Test
    fun testForward() {

        val actual = expandForConvolution(floatArrayOf(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f), 3, 2, 1, 2, 3)
        val expected = floatMatrixFromColumns(
            floatArrayOf(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f),
            floatArrayOf(4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f)
        )

        assertMatrixEquality(expected, actual, 0.01f)

    }

    @Test
    fun testFirstColumnOfConvolution() {

        /*
            *1* 2
             3  4
         */
        assertEquals(0, firstColumnOfConvolution(0, 2))

        /*
            1 *2*
            3  4
         */
        assertEquals(1, firstColumnOfConvolution(1, 2))

        /*
            1  2
           *3* 4
        */
        assertEquals(0, firstColumnOfConvolution(2, 2))

        /*
           1  2
           3 *4*
        */
        assertEquals(1, firstColumnOfConvolution(3, 2))

    }

    @Test
    fun testFirstRowOfConvolution() {

        /*
            *1* 2
             3  4
         */
        assertEquals(0, firstRowOfConvolution(0, 2))

        /*
            1 *2*
            3  4
         */
        assertEquals(0, firstRowOfConvolution(1, 2))

        /*
            1  2
           *3* 4
        */
        assertEquals(1, firstRowOfConvolution(2, 2))

        /*
           1  2
           3 *4*
        */
        assertEquals(1, firstRowOfConvolution(3, 2))

    }

}
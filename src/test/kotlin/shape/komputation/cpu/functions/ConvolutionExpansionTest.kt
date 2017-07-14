package shape.komputation.cpu.functions

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test
import shape.komputation.assertMatrixEquality
import shape.komputation.matrix.doubleMatrixFromColumns

class ConvolutionExpansionTest {

    @Test
    fun testForward() {

        val actual = expandForConvolution(doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0), 3, 2, 1, 2, 3)
        val expected = doubleMatrixFromColumns(
            doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
            doubleArrayOf(4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
        )

        assertMatrixEquality(expected, actual, 0.01)

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
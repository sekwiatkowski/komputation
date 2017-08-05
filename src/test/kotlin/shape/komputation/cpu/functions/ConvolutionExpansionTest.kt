package shape.komputation.cpu.functions

import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test
import shape.komputation.assertMatrixEquality
import shape.komputation.matrix.floatMatrixFromColumns

class ConvolutionExpansionTest {

    @Test
    fun testForward() {
        /*
            1 4 7
            2 5 8
            3 6 9
         */

        val input = floatArrayOf(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f)

        val size = 2 * 3 * 2
        val actual = FloatArray(size)
        expandForConvolution(input, 3, actual, size, 1, 2, 2, 3)

        val expected = floatArrayOf(
            1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
            4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f
        )

        assertArrayEquals(expected, actual, 0.01f)

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
        assertEquals(0, firstColumnOfConvolution(1, 2))

        /*
            1  2
           *3* 4
        */
        assertEquals(1, firstColumnOfConvolution(2, 2))

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
        assertEquals(1, firstRowOfConvolution(1, 2))

        /*
            1  2
           *3* 4
        */
        assertEquals(0, firstRowOfConvolution(2, 2))

        /*
           1  2
           3 *4*
        */
        assertEquals(1, firstRowOfConvolution(3, 2))

    }

}
package shape.komputation.cpu.functions

import org.junit.jupiter.api.Test
import shape.komputation.assertMatrixEquality
import shape.komputation.matrix.floatMatrixFromRows

class RowSplitTest {

    val firstRow = floatArrayOf(1.0f, 2.0f)
    val secondRow = floatArrayOf(3.0f, 4.0f)
    val thirdRow = floatArrayOf(5.0f, 6.0f)

    val matrix = floatMatrixFromRows(
        firstRow,
        secondRow,
        thirdRow
    )

    @Test
    fun test1() {

        val (firstActual, secondActual) = splitRows(
            matrix,
            intArrayOf(1, 2))

        val firstExpected = floatMatrixFromRows(
            firstRow
        )

        val secondExpected = floatMatrixFromRows(
            secondRow,
            thirdRow
        )

        assertMatrixEquality(firstExpected, firstActual, 0.01f)
        assertMatrixEquality(secondExpected, secondActual, 0.01f)

    }

    @Test
    fun test2() {

        val (firstActual, secondActual) = splitRows(matrix, intArrayOf(2, 1))

        val firstExpected = floatMatrixFromRows(
            firstRow,
            secondRow
        )

        val secondExpected = floatMatrixFromRows(
            thirdRow
        )

        assertMatrixEquality(firstExpected, firstActual, 0.01f)
        assertMatrixEquality(secondExpected, secondActual, 0.01f)

    }

}
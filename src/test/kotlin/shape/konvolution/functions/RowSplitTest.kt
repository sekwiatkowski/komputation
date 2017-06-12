package shape.konvolution.functions

import org.junit.jupiter.api.Test
import shape.konvolution.assertMatrixEquality
import shape.konvolution.matrix.createRealMatrix
import shape.konvolution.matrix.createRealVector

class RowSplitTest {

    val firstRow = doubleArrayOf(1.0, 2.0)
    val secondRow = doubleArrayOf(3.0, 4.0)
    val thirdRow = doubleArrayOf(5.0, 6.0)

    val matrix = createRealMatrix(
        firstRow,
        secondRow,
        thirdRow
    )

    @Test
    fun test1() {

        val (firstActual, secondActual) = splitRows(matrix, intArrayOf(1, 2))

        val firstExpected = createRealMatrix(
            firstRow
        )

        val secondExpected = createRealMatrix(
            secondRow,
            thirdRow
        )

        assertMatrixEquality(firstExpected, firstActual, 0.01)
        assertMatrixEquality(secondExpected, secondActual, 0.01)

    }

    @Test
    fun test2() {

        val (firstActual, secondActual) = splitRows(matrix, intArrayOf(2, 1))

        val firstExpected = createRealMatrix(
            firstRow,
            secondRow
        )

        val secondExpected = createRealMatrix(
            thirdRow
        )

        assertMatrixEquality(firstExpected, firstActual, 0.01)
        assertMatrixEquality(secondExpected, secondActual, 0.01)

    }

}
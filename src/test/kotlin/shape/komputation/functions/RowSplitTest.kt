package shape.komputation.functions

import org.junit.jupiter.api.Test
import shape.komputation.assertMatrixEquality
import shape.komputation.matrix.doubleColumnVector
import shape.komputation.matrix.doubleRowMatrix

class RowSplitTest {

    val firstRow = doubleColumnVector(1.0, 2.0)
    val secondRow = doubleColumnVector(3.0, 4.0)
    val thirdRow = doubleColumnVector(5.0, 6.0)

    val matrix = doubleRowMatrix(
        firstRow,
        secondRow,
        thirdRow
    )

    @Test
    fun test1() {

        val (firstActual, secondActual) = splitRows(
            matrix,
            intArrayOf(1, 2))

        val firstExpected = doubleRowMatrix(
            firstRow
        )

        val secondExpected = doubleRowMatrix(
            secondRow,
            thirdRow
        )

        assertMatrixEquality(firstExpected, firstActual, 0.01)
        assertMatrixEquality(secondExpected, secondActual, 0.01)

    }

    @Test
    fun test2() {

        val (firstActual, secondActual) = splitRows(matrix, intArrayOf(2, 1))

        val firstExpected = doubleRowMatrix(
            firstRow,
            secondRow
        )

        val secondExpected = doubleRowMatrix(
            thirdRow
        )

        assertMatrixEquality(firstExpected, firstActual, 0.01)
        assertMatrixEquality(secondExpected, secondActual, 0.01)

    }

}
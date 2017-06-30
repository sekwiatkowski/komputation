package shape.komputation.functions

import org.junit.jupiter.api.Test
import shape.komputation.assertMatrixEquality
import shape.komputation.matrix.doubleMatrixFromRows
import shape.komputation.matrix.doubleRowVector

class RowStackingTest {

    @Test
    fun test1() {

        val firstRow = doubleArrayOf(1.0, 2.0)
        val secondRow = doubleArrayOf(3.0, 4.0)

        val actual = stackRows(2, doubleRowVector(*firstRow), doubleRowVector(*secondRow))
        val expected = doubleMatrixFromRows(
            firstRow,
            secondRow
        )

        assertMatrixEquality(expected, actual, 0.01)

    }

    @Test
    fun test2() {

        val firstRow = doubleArrayOf(1.0, 2.0)

        val secondRow = doubleArrayOf(3.0, 4.0)
        val thirdRow = doubleArrayOf(5.0, 6.0)

        val secondMatrix = doubleMatrixFromRows(
            secondRow,
            thirdRow
        )

        val actual = stackRows(2, doubleRowVector(*firstRow), secondMatrix)
        val expected = doubleMatrixFromRows(
            firstRow,
            secondRow,
            thirdRow
        )

        assertMatrixEquality(expected, actual, 0.01)

    }

}
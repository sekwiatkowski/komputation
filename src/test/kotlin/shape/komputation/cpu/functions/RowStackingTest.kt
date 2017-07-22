package shape.komputation.cpu.functions

import org.junit.jupiter.api.Test
import shape.komputation.assertMatrixEquality
import shape.komputation.matrix.floatMatrixFromRows
import shape.komputation.matrix.floatRowVector

class RowStackingTest {

    @Test
    fun test1() {

        val firstRow = floatArrayOf(1.0f, 2.0f)
        val secondRow = floatArrayOf(3.0f, 4.0f)

        val actual = stackRows(2, floatRowVector(*firstRow), floatRowVector(*secondRow))
        val expected = floatMatrixFromRows(
            firstRow,
            secondRow
        )

        assertMatrixEquality(expected, actual, 0.01f)

    }

    @Test
    fun test2() {

        val firstRow = floatArrayOf(1.0f, 2.0f)

        val secondRow = floatArrayOf(3.0f, 4.0f)
        val thirdRow = floatArrayOf(5.0f, 6.0f)

        val secondMatrix = floatMatrixFromRows(
            secondRow,
            thirdRow
        )

        val actual = stackRows(2, floatRowVector(*firstRow), secondMatrix)
        val expected = floatMatrixFromRows(
            firstRow,
            secondRow,
            thirdRow
        )

        assertMatrixEquality(expected, actual, 0.01f)

    }

}
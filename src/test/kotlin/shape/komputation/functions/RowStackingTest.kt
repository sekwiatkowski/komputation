package shape.komputation.functions

import org.junit.jupiter.api.Test
import shape.komputation.assertMatrixEquality
import shape.komputation.matrix.doubleRowVector
import shape.komputation.matrix.doubleMatrixFromRows

class RowStackingTest {

    @Test
    fun test1() {

        val firstRowVector = doubleRowVector(1.0, 2.0)
        val secondRowVector = doubleRowVector(3.0, 4.0)

        val actual = stackRows(2, firstRowVector, secondRowVector)
        val expected = doubleMatrixFromRows(
            firstRowVector,
            secondRowVector
        )

        assertMatrixEquality(expected, actual, 0.01)

    }

    @Test
    fun test2() {

        val firstRowVector = doubleRowVector(1.0, 2.0)

        val secondRowVector = doubleRowVector(3.0, 4.0)
        val thirdRowVector = doubleRowVector(5.0, 6.0)

        val secondMatrix = doubleMatrixFromRows(
            secondRowVector,
            thirdRowVector
        )

        val actual = stackRows(2, firstRowVector, secondMatrix)
        val expected = doubleMatrixFromRows(
            firstRowVector,
            secondRowVector,
            thirdRowVector
        )

        assertMatrixEquality(expected, actual, 0.01)

    }

}
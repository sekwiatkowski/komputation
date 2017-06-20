package shape.komputation.functions

import org.junit.jupiter.api.Test
import shape.komputation.assertMatrixEquality
import shape.komputation.matrix.doubleRowVector
import shape.komputation.matrix.doubleMatrixFromRows

class RowConcatenationTest {

    @Test
    fun test1() {

        val firstColumnVector = doubleRowVector(1.0, 2.0)
        val secondColumnVector = doubleRowVector(3.0, 4.0)

        val actual = concatRows(firstColumnVector, secondColumnVector)
        val expected = doubleMatrixFromRows(
            firstColumnVector,
            secondColumnVector
        )

        assertMatrixEquality(expected, actual, 0.01)

    }

    @Test
    fun test2() {

        val firstVector = doubleRowVector(1.0, 2.0)

        val secondVector = doubleRowVector(3.0, 4.0)
        val thirdVector = doubleRowVector(5.0, 6.0)

        val secondMatrix = doubleMatrixFromRows(
            secondVector,
            thirdVector
        )

        val actual = concatRows(firstVector, secondMatrix)
        val expected = doubleMatrixFromRows(
            firstVector,
            secondVector,
            thirdVector
        )

        assertMatrixEquality(expected, actual, 0.01)

    }

}
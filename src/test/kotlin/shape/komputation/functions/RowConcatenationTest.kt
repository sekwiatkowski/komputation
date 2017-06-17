package shape.komputation.functions

import org.junit.jupiter.api.Test
import shape.komputation.assertMatrixEquality
import shape.komputation.matrix.doubleColumnVector
import shape.komputation.matrix.doubleRowMatrix

class RowConcatenationTest {

    @Test
    fun test1() {

        val firstColumnVector = doubleColumnVector(1.0, 2.0)
        val secondColumnVector = doubleColumnVector(3.0, 4.0)

        val actual = concatRows(firstColumnVector, secondColumnVector)
        val expected = doubleRowMatrix(
            firstColumnVector,
            secondColumnVector
        )

        assertMatrixEquality(expected, actual, 0.01)

    }

    @Test
    fun test2() {

        val firstVector = doubleColumnVector(1.0, 2.0)

        val secondVector = doubleColumnVector(3.0, 4.0)
        val thirdVector = doubleColumnVector(5.0, 6.0)

        val secondMatrix = doubleRowMatrix(
            secondVector,
            thirdVector
        )

        val actual = concatRows(firstVector, secondMatrix)
        val expected = doubleRowMatrix(
            firstVector,
            secondVector,
            thirdVector
        )

        assertMatrixEquality(expected, actual, 0.01)

    }

}
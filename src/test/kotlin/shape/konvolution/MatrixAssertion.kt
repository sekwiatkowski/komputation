package shape.konvolution

import org.junit.jupiter.api.Assertions.assertEquals

fun assertMatrixEquality(expected: RealMatrix, actual: RealMatrix, delta : Double) {

    assertEquals(expected.numberRows(), actual.numberRows())
    assertEquals(expected.numberColumns(), actual.numberColumns())

    for (indexRow in 0..actual.numberRows() - 1) {

        for (indexColumn in 0..actual.numberColumns() - 1) {

            assertEquals(expected.get(indexRow, indexColumn), actual.get(indexRow, indexColumn), delta)
        }

    }
}

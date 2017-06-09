package shape.konvolution

import no.uib.cipr.matrix.Matrix
import org.junit.jupiter.api.Assertions.assertEquals

fun assertMatrixEquality(expected: Matrix, actual: Matrix, delta : Double) {

    assertEquals(expected.numRows(), actual.numRows())
    assertEquals(expected.numColumns(), actual.numColumns())

    for (indexRow in 0..actual.numRows() - 1) {

        for (indexColumn in 0..actual.numColumns() - 1) {

            assertEquals(expected.get(indexRow, indexColumn), actual.get(indexRow, indexColumn), delta)
        }

    }
}

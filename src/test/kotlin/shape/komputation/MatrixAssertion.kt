package shape.komputation

import org.junit.jupiter.api.Assertions.assertEquals
import shape.komputation.matrix.DoubleMatrix

fun assertMatrixEquality(expected: DoubleMatrix, actual: DoubleMatrix, delta : Double) {

    assertEquals(expected.numberRows, actual.numberRows)
    assertEquals(expected.numberColumns, actual.numberColumns)

    val expectedEntries = expected.entries
    val actualEntries = actual.entries

    for (index in 0..actualEntries.size - 1) {

        assertEquals(actualEntries[index], expectedEntries[index], delta)
    }

}

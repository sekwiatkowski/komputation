package shape.komputation

import org.junit.jupiter.api.Assertions.assertEquals
import shape.komputation.matrix.FloatMatrix

fun assertMatrixEquality(expected: FloatMatrix, actual: FloatMatrix, delta : Float) {

    assertEquals(expected.numberRows, actual.numberRows)
    assertEquals(expected.numberColumns, actual.numberColumns)

    val expectedEntries = expected.entries
    val actualEntries = actual.entries

    for (index in 0..actualEntries.size - 1) {

        assertEquals(actualEntries[index], expectedEntries[index], delta)
    }

}

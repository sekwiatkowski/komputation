package shape.komputation.matrix

import org.junit.jupiter.api.Test
import org.junit.jupiter.api.Assertions.assertArrayEquals
import kotlin.test.assertEquals

class SequenceMatrixTest {

    @Test
    fun testSequence() {

        val actual = sequence(
            2,
            step(1.0, 0.0),
            step(0.0, 1.0)
        )

        assertArrayEquals(actual.entries, doubleArrayOf(1.0, 0.0, 0.0, 1.0))
        assertEquals(actual.numberSteps, 2)
        assertEquals(actual.numberColumns, 2)
        assertEquals(actual.numberRows, 2)
        assertEquals(actual.numberColumns, 2)

    }

    @Test
    fun testSetStep() {

        val matrix = zeroSequenceMatrix(2, 2, 1)

        val firstStep = doubleArrayOf(1.1, 1.2)
        val secondStep = doubleArrayOf(2.1, 2.2)

        matrix.setStep(0, firstStep)
        matrix.setStep(1, secondStep)

        assertArrayEquals(matrix.getStep(0).entries, firstStep)
        assertArrayEquals(matrix.getStep(1).entries, secondStep)

    }


}
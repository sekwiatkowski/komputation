package shape.komputation.functions.activation

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

class TanhTest {

    @Test
    fun test1() {

        assertEquals(0.76159415595, tanh(1.0), 0.0001)

    }

    @Test
    fun test2() {

        assertEquals(0.96402758007, tanh(2.0), 0.0001)

    }


}
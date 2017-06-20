package shape.komputation.matrix

import org.junit.jupiter.api.Test
import org.junit.jupiter.api.Assertions.assertArrayEquals

class PartitioningTest {

    @Test
    fun test1() {

        val actual = partitionIndices(2, 1)
        val expected = arrayOf(arrayOf(0), arrayOf(1))

        assertArrayEquals(expected, actual)

    }


    @Test
    fun test2() {

        val actual = partitionIndices(2, 2)
        val expected = arrayOf(arrayOf(0, 1))

        assertArrayEquals(expected, actual)

    }

    @Test
    fun test3() {

        val actual = partitionIndices(2, 3)
        val expected = arrayOf(arrayOf(0, 1))

        assertArrayEquals(expected, actual)

    }

}
package shape.konvolution.optimization

import org.junit.jupiter.api.Test
import shape.konvolution.assertMatrixEquality
import shape.konvolution.matrix.createRealMatrix

class SparseUpdateTest {

    @Test
    fun test1() {

        val actual = arrayOf(
            doubleArrayOf(1.0, 2.0),
            doubleArrayOf(3.0, 4.0)
        )

        val updates = createRealMatrix(
            doubleArrayOf(1.1, 2.1)
        )

        optimizeSparsely(actual, intArrayOf(0), updates)

        val expected = arrayOf(
            doubleArrayOf(1.1, 2.1),
            doubleArrayOf(3.0, 4.0)
        )

        assertMatrixEquality(createRealMatrix(*expected), createRealMatrix(*actual), 0.01)

    }

    @Test
    fun test2() {

        val actual = arrayOf(
            doubleArrayOf(1.0, 2.0),
            doubleArrayOf(3.0, 4.0)
        )

        val updates = createRealMatrix(
            doubleArrayOf(3.1, 4.1)
        )

        optimizeSparsely(actual, intArrayOf(1), updates)

        val expected = arrayOf(
            doubleArrayOf(1.0, 2.0),
            doubleArrayOf(3.1, 4.1)
        )

        assertMatrixEquality(createRealMatrix(*expected), createRealMatrix(*actual), 0.01)

    }

}
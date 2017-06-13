package shape.komputation.layers.entry

import org.junit.jupiter.api.Test
import shape.komputation.assertMatrixEquality
import shape.komputation.matrix.createIntegerVector
import shape.komputation.matrix.createRealMatrix

class LookupLayerTest {

    val firstEmbedding = doubleArrayOf(1.1, 1.2)
    val secondEmbedding = doubleArrayOf(2.1, 2.2)

    val embeddings = arrayOf(
        firstEmbedding,
        secondEmbedding
    )

    val lookupLayer = createLookupLayer(embeddings)

    @Test
    fun testForward1() {

        val expected = createRealMatrix(firstEmbedding)

        val actual = lookupLayer.forward(createIntegerVector(0))

        assertMatrixEquality(expected, actual, 0.001)

    }

    @Test
    fun testForward2() {

        val expected = createRealMatrix(secondEmbedding)

        val actual = lookupLayer.forward(createIntegerVector(1))

        assertMatrixEquality(expected, actual, 0.001)

    }

    @Test
    fun testForward3() {

        val expected = createRealMatrix(firstEmbedding, secondEmbedding)

        val actual = lookupLayer.forward(createIntegerVector(0, 1))

        assertMatrixEquality(expected, actual, 0.001)

    }

    @Test
    fun testForward4() {

        val expected = createRealMatrix(secondEmbedding, firstEmbedding)

        val actual = lookupLayer.forward(createIntegerVector(1, 0))

        assertMatrixEquality(expected, actual, 0.001)

    }


}
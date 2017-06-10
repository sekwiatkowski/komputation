package shape.konvolution.layers.entry

import org.junit.jupiter.api.Test
import shape.konvolution.assertMatrixEquality
import shape.konvolution.matrix.createIntegerVector
import shape.konvolution.matrix.createRealMatrix

class LookupLayerTest {

    val firstEmbedding = doubleArrayOf(1.1, 1.2)
    val secondEmbedding = doubleArrayOf(2.1, 2.2)

    val embeddings = arrayOf(
        firstEmbedding,
        secondEmbedding
    )

    val embeddingLayer = LookupLayer(embeddings)

    @Test
    fun testForward1() {

        val actual = embeddingLayer.forward(createIntegerVector(0))
        val expected = createRealMatrix(firstEmbedding)

        assertMatrixEquality(expected, actual, 0.001)

    }

    @Test
    fun testForward2() {

        val actual = embeddingLayer.forward(createIntegerVector(1))
        val expected = createRealMatrix(secondEmbedding)

        assertMatrixEquality(expected, actual, 0.001)

    }

    @Test
    fun testForward3() {

        val actual = embeddingLayer.forward(createIntegerVector(0, 1))
        val expected = createRealMatrix(firstEmbedding, secondEmbedding)

        assertMatrixEquality(expected, actual, 0.001)

    }

    @Test
    fun testForward4() {

        val actual = embeddingLayer.forward(createIntegerVector(1, 0))
        val expected = createRealMatrix(secondEmbedding, firstEmbedding)

        assertMatrixEquality(expected, actual, 0.001)

    }


}
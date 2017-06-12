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

    val embeddingLayer = createLookupLayer(embeddings)

    @Test
    fun testForward1() {

        embeddingLayer.setInput(createIntegerVector(0))
        embeddingLayer.forward()

        val expected = createRealMatrix(firstEmbedding)

        val actual = embeddingLayer.lastForwardResult!!

        assertMatrixEquality(expected, actual, 0.001)

    }

    @Test
    fun testForward2() {

        embeddingLayer.setInput(createIntegerVector(1))
        embeddingLayer.forward()

        val expected = createRealMatrix(secondEmbedding)

        val actual = embeddingLayer.lastForwardResult!!

        assertMatrixEquality(expected, actual, 0.001)

    }

    @Test
    fun testForward3() {

        embeddingLayer.setInput(createIntegerVector(0, 1))
        embeddingLayer.forward()

        val expected = createRealMatrix(firstEmbedding, secondEmbedding)

        val actual = embeddingLayer.lastForwardResult!!

        assertMatrixEquality(expected, actual, 0.001)

    }

    @Test
    fun testForward4() {

        embeddingLayer.setInput(createIntegerVector(1, 0))
        embeddingLayer.forward()

        val expected = createRealMatrix(secondEmbedding, firstEmbedding)

        val actual = embeddingLayer.lastForwardResult!!

        assertMatrixEquality(expected, actual, 0.001)

    }


}
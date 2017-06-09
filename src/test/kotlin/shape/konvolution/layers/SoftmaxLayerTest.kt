package shape.konvolution.layers

import org.junit.jupiter.api.Test
import shape.konvolution.assertMatrixEquality
import shape.konvolution.createDenseMatrix

class SoftmaxLayerTest {

    @Test
    fun test() {

        val softmaxLayer = SoftmaxLayer()

        val input = createDenseMatrix(doubleArrayOf(1.0, 2.0), doubleArrayOf(3.0, 4.0))

        val actual = softmaxLayer.forward(input)
        val expected = createDenseMatrix(
            doubleArrayOf(Math.exp(1.0)/(Math.exp(1.0)+Math.exp(3.0)), Math.exp(2.0)/(Math.exp(2.0)+Math.exp(4.0))),
            doubleArrayOf(Math.exp(3.0)/(Math.exp(1.0)+Math.exp(3.0)), Math.exp(4.0)/(Math.exp(2.0)+Math.exp(4.0)))
        )

        assertMatrixEquality(expected, actual, 0.0001)

    }


}
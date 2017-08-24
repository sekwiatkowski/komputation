package shape.komputation.cuda.layers.entry

import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Test
import shape.komputation.cuda.getFloatArray
import shape.komputation.cuda.memory.InputMemory
import shape.komputation.cuda.setUpCudaContext
import shape.komputation.layers.entry.lookupLayer
import shape.komputation.matrix.Matrix
import shape.komputation.matrix.intMatrix

class CudaLookupLayerTest {

    @Test
    fun testOneVectorOneDimension() {

        val vectors = arrayOf(floatArrayOf(1f))
        val dimension = 1
        val batch = intArrayOf(0)
        val maximumLength = 1
        val inputs : Array<Matrix> = arrayOf(intMatrix(0))
        val expected = floatArrayOf(1f)

        test(vectors, dimension, batch, true, maximumLength, inputs, expected)

    }

    @Test
    fun testOneVectorTwoDimensions() {

        val vectors = arrayOf(floatArrayOf(1f, 2f))
        val dimension = 2
        val batch = intArrayOf(0)
        val maximumLength = 1
        val inputs : Array<Matrix> = arrayOf(intMatrix(0))
        val expected = floatArrayOf(1f, 2f)

        test(vectors, dimension, batch, true, maximumLength, inputs, expected)

    }

    @Test
    fun testTwoVectorsOneDimension() {

        val vectors = arrayOf(floatArrayOf(1f), floatArrayOf(2f))
        val dimension = 1
        val batch = intArrayOf(0)
        val maximumLength = 2
        val inputs : Array<Matrix> = arrayOf(intMatrix(0, 1))
        val expected = floatArrayOf(1f, 2f)

        test(vectors, dimension, batch, true, maximumLength, inputs, expected)

    }

    @Test
    fun testTwoVectorsOneDimensionReversed() {

        val vectors = arrayOf(floatArrayOf(1f), floatArrayOf(2f))
        val dimension = 1
        val batch = intArrayOf(0)
        val maximumLength = 2
        val inputs : Array<Matrix> = arrayOf(intMatrix(1, 0))
        val expected = floatArrayOf(2f, 1f)

        test(vectors, dimension, batch, false, maximumLength, inputs, expected)

    }

    @Test
    fun testFirstOfTwoVectorsOneDimension() {

        val vectors = arrayOf(floatArrayOf(1f), floatArrayOf(2f))
        val dimension = 1
        val batch = intArrayOf(0)
        val maximumLength = 2
        val inputs : Array<Matrix> = arrayOf(intMatrix(0))
        val expected = floatArrayOf(1f, Float.NaN)

        test(vectors, dimension, batch, false, maximumLength, inputs, expected)

    }

    @Test
    fun testLastOfTwoVectorsOneDimension() {

        val vectors = arrayOf(floatArrayOf(1f), floatArrayOf(2f))
        val dimension = 1
        val batch = intArrayOf(0)
        val maximumLength = 2
        val inputs : Array<Matrix> = arrayOf(intMatrix(1))
        val expected = floatArrayOf(2f, Float.NaN)

        test(vectors, dimension, batch, false, maximumLength, inputs, expected)

    }

    @Test
    fun testTwoOutOfThreeVectorsTwoDimensions() {

        val vectors = arrayOf(floatArrayOf(1f, 2f), floatArrayOf(3f, 4f), floatArrayOf(5f, 6f))
        val dimension = 2
        val batch = intArrayOf(0)
        val maximumLength = 2
        val inputs : Array<Matrix> = arrayOf(intMatrix(2, 0))
        val expected = floatArrayOf(5f, 6f, 1f, 2f)

        test(vectors, dimension, batch, true, maximumLength, inputs, expected)

    }

    @Test
    fun testTwoInstancesOneVectorOneDimension() {

        val vectors = arrayOf(floatArrayOf(1f))
        val dimension = 1
        val batch = intArrayOf(0, 1)
        val maximumLength = 1
        val inputs : Array<Matrix> = arrayOf(intMatrix(0), intMatrix(0))
        val expected = floatArrayOf(1f, 1f)

        test(vectors, dimension, batch, true, maximumLength, inputs, expected)

    }

    @Test
    fun testTwoInstancesTwoVectorsOneDimension() {

        val vectors = arrayOf(floatArrayOf(1f), floatArrayOf(2f))
        val dimension = 1
        val batch = intArrayOf(0, 1)
        val maximumLength = 1
        val inputs : Array<Matrix> = arrayOf(intMatrix(0), intMatrix(1))
        val expected = floatArrayOf(1f, 2f)

        test(vectors, dimension, batch, true, maximumLength, inputs, expected)

    }

    private fun test(vectors: Array<FloatArray>, dimension : Int, batch: IntArray, hasFixedLength : Boolean, maximumLength: Int, inputs: Array<Matrix>, expected: FloatArray) {

        val cudaContext = setUpCudaContext()

        val lookupLayer = lookupLayer(vectors, maximumLength, hasFixedLength, dimension).buildForCuda(cudaContext)

        val maximumBatchSize = batch.size

        lookupLayer.acquire(maximumBatchSize)

        val deviceResult = lookupLayer.forward(0, maximumBatchSize, batch, inputs, InputMemory())

        val actual = getFloatArray(deviceResult, batch.size * maximumLength * dimension)

        lookupLayer.release()

        assertArrayEquals(expected, actual)

    }

}
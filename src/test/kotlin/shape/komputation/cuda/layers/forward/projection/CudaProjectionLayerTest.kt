package shape.komputation.cuda.layers.forward.projection

import jcuda.Pointer
import jcuda.jcublas.cublasHandle
import jcuda.runtime.JCuda.cudaFree
import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Test
import shape.komputation.cuda.getVector
import shape.komputation.cuda.setUpCudaContext
import shape.komputation.cuda.setVector
import shape.komputation.initialization.providedInitialization
import shape.komputation.layers.forward.projection.projectionLayer

class CudaProjectionLayerTest {

    @Test
    fun testOneDimensionToOneDimension() {

        val input = doubleArrayOf(2.0)
        val weights = doubleArrayOf(3.0)
        val expected = doubleArrayOf(6.0)
        val inputDimension = 1
        val outputDimension = 1

        test(inputDimension, outputDimension, input, weights, expected)

    }

    @Test
    fun testOneDimensionToTwoDimensions() {

        val input = doubleArrayOf(2.0)
        val weights = doubleArrayOf(3.0, 4.0)
        val expected = doubleArrayOf(6.0, 8.0)
        val inputDimension = 1
        val outputDimension = 2

        test(inputDimension, outputDimension, input, weights, expected)

    }

    @Test
    fun testTwoDimensionsToOneDimensions() {

        val input = doubleArrayOf(2.0, 3.0)
        val weights = doubleArrayOf(4.0, 5.0)
        val expected = doubleArrayOf(23.0)
        val inputDimension = 2
        val outputDimension = 1

        test(inputDimension, outputDimension, input, weights, expected)

    }

    @Test
    fun testTwoDimensionsToTwoDimensions() {

        /*
                              2
                              3
            4 6    4*2+6*3 = 26
            5 7    5*2+7*3 = 31
         */
        val input = doubleArrayOf(2.0, 3.0)
        val weights = doubleArrayOf(4.0, 5.0, 6.0, 7.0)
        val expected = doubleArrayOf(26.0, 31.0)
        val inputDimension = 2
        val outputDimension = 2

        test(inputDimension, outputDimension, input, weights, expected)

    }

    private fun test(inputDimension: Int, outputDimension: Int, input: DoubleArray, weights : DoubleArray, expected: DoubleArray) {

        val context = setUpCudaContext()

        val projectionLayer = projectionLayer(inputDimension, outputDimension, providedInitialization(weights, outputDimension))
            .buildForCuda(context, cublasHandle())

        projectionLayer.acquire()

        val deviceInput = Pointer()
        setVector(input, inputDimension, deviceInput)

        val deviceResult = projectionLayer.forward(deviceInput)

        val actual = getVector(deviceResult, outputDimension)

        projectionLayer.release()

        cudaFree(deviceInput)

        context.destroy()

        assertArrayEquals(expected, actual)

    }

}

class CudaProjectionLayerWithBiasTest {

    @Test
    fun testOneDimensionToOneDimension() {

        val input = doubleArrayOf(2.0)
        val weights = doubleArrayOf(3.0)
        val bias = doubleArrayOf(4.0)
        val expected = doubleArrayOf(10.0)
        val inputDimension = 1
        val outputDimension = 1

        test(inputDimension, outputDimension, input, weights, bias, expected)

    }

    @Test
    fun testOneDimensionToTwoDimensions() {

        val input = doubleArrayOf(2.0)
        val weights = doubleArrayOf(3.0, 4.0)
        val bias = doubleArrayOf(5.0, 6.0)
        val expected = doubleArrayOf(11.0, 14.0)
        val inputDimension = 1
        val outputDimension = 2

        test(inputDimension, outputDimension, input, weights, bias, expected)

    }

    @Test
    fun testTwoDimensionsToOneDimensions() {

        val input = doubleArrayOf(2.0, 3.0)
        val weights = doubleArrayOf(4.0, 5.0)
        val bias = doubleArrayOf(6.0)
        val expected = doubleArrayOf(29.0)
        val inputDimension = 2
        val outputDimension = 1

        test(inputDimension, outputDimension, input, weights, bias, expected)

    }

    @Test
    fun testTwoDimensionsToTwoDimensions() {

        val input = doubleArrayOf(2.0, 3.0)
        val weights = doubleArrayOf(4.0, 5.0, 6.0, 7.0)
        val bias = doubleArrayOf(8.0, 9.0)
        val expected = doubleArrayOf(34.0, 40.0)
        val inputDimension = 2
        val outputDimension = 2

        test(inputDimension, outputDimension, input, weights, bias, expected)

    }

    private fun test(inputDimension: Int, outputDimension: Int, input: DoubleArray, weights : DoubleArray, bias : DoubleArray, expected: DoubleArray) {

        val context = setUpCudaContext()

        val projectionLayer = projectionLayer(inputDimension, outputDimension, providedInitialization(weights, outputDimension), providedInitialization(bias, outputDimension))
            .buildForCuda(context, cublasHandle())

        projectionLayer.acquire()

        val deviceInput = Pointer()
        setVector(input, inputDimension, deviceInput)

        val deviceResult = projectionLayer.forward(deviceInput)

        val actual = getVector(deviceResult, outputDimension)

        projectionLayer.release()

        cudaFree(deviceInput)

        context.destroy()

        assertArrayEquals(actual, expected)

    }

}
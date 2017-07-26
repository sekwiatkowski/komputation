package shape.komputation.cuda.layers.forward.projection

import jcuda.Pointer
import jcuda.jcublas.cublasHandle
import jcuda.runtime.JCuda.cudaFree
import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Test
import shape.komputation.cuda.getFloatArray
import shape.komputation.cuda.setUpCudaContext
import shape.komputation.cuda.setFloatArray
import shape.komputation.initialization.providedInitialization
import shape.komputation.layers.forward.projection.projectionLayer

class CudaProjectionLayerTest {

    @Test
    fun testOneDimensionToOneDimension() {

        val input = floatArrayOf(2.0f)
        val weights = floatArrayOf(3.0f)
        val expected = floatArrayOf(6.0f)
        val inputDimension = 1
        val outputDimension = 1

        test(inputDimension, outputDimension, input, weights, expected)

    }

    @Test
    fun testOneDimensionToTwoDimensions() {

        val input = floatArrayOf(2.0f)
        val weights = floatArrayOf(3.0f, 4.0f)
        val expected = floatArrayOf(6.0f, 8.0f)
        val inputDimension = 1
        val outputDimension = 2

        test(inputDimension, outputDimension, input, weights, expected)

    }

    @Test
    fun testTwoDimensionsToOneDimensions() {

        val input = floatArrayOf(2.0f, 3.0f)
        val weights = floatArrayOf(4.0f, 5.0f)
        val expected = floatArrayOf(23.0f)
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
        val input = floatArrayOf(2.0f, 3.0f)
        val weights = floatArrayOf(4.0f, 5.0f, 6.0f, 7.0f)
        val expected = floatArrayOf(26.0f, 31.0f)
        val inputDimension = 2
        val outputDimension = 2

        test(inputDimension, outputDimension, input, weights, expected)

    }

    private fun test(inputDimension: Int, outputDimension: Int, input: FloatArray, weights : FloatArray, expected: FloatArray) {

        val context = setUpCudaContext()

        val projectionLayer = projectionLayer(inputDimension, outputDimension, providedInitialization(weights, outputDimension))
            .buildForCuda(context, cublasHandle())

        projectionLayer.acquire()

        val deviceInput = Pointer()
        setFloatArray(input, inputDimension, deviceInput)

        val deviceResult = projectionLayer.forward(deviceInput, true)

        val actual = getFloatArray(deviceResult, outputDimension)

        projectionLayer.release()

        cudaFree(deviceInput)

        context.destroy()

        assertArrayEquals(expected, actual)

    }

}

class CudaProjectionLayerWithBiasTest {

    @Test
    fun testOneDimensionToOneDimension() {

        val input = floatArrayOf(2.0f)
        val weights = floatArrayOf(3.0f)
        val bias = floatArrayOf(4.0f)
        val expected = floatArrayOf(10.0f)
        val inputDimension = 1
        val outputDimension = 1

        test(inputDimension, outputDimension, input, weights, bias, expected)

    }

    @Test
    fun testOneDimensionToTwoDimensions() {

        val input = floatArrayOf(2.0f)
        val weights = floatArrayOf(3.0f, 4.0f)
        val bias = floatArrayOf(5.0f, 6.0f)
        val expected = floatArrayOf(11.0f, 14.0f)
        val inputDimension = 1
        val outputDimension = 2

        test(inputDimension, outputDimension, input, weights, bias, expected)

    }

    @Test
    fun testTwoDimensionsToOneDimensions() {

        val input = floatArrayOf(2.0f, 3.0f)
        val weights = floatArrayOf(4.0f, 5.0f)
        val bias = floatArrayOf(6.0f)
        val expected = floatArrayOf(29.0f)
        val inputDimension = 2
        val outputDimension = 1

        test(inputDimension, outputDimension, input, weights, bias, expected)

    }

    @Test
    fun testTwoDimensionsToTwoDimensions() {

        val input = floatArrayOf(2.0f, 3.0f)
        val weights = floatArrayOf(4.0f, 5.0f, 6.0f, 7.0f)
        val bias = floatArrayOf(8.0f, 9.0f)
        val expected = floatArrayOf(34.0f, 40.0f)
        val inputDimension = 2
        val outputDimension = 2

        test(inputDimension, outputDimension, input, weights, bias, expected)

    }

    private fun test(inputDimension: Int, outputDimension: Int, input: FloatArray, weights : FloatArray, bias : FloatArray, expected: FloatArray) {

        val context = setUpCudaContext()

        val projectionLayer = projectionLayer(inputDimension, outputDimension, providedInitialization(weights, outputDimension), providedInitialization(bias, outputDimension))
            .buildForCuda(context, cublasHandle())

        projectionLayer.acquire()

        val deviceInput = Pointer()
        setFloatArray(input, inputDimension, deviceInput)

        val deviceResult = projectionLayer.forward(deviceInput, true)

        val actual = getFloatArray(deviceResult, outputDimension)

        projectionLayer.release()

        cudaFree(deviceInput)

        context.destroy()

        assertArrayEquals(actual, expected)

    }

}
package shape.komputation.cuda.functions

import jcuda.Pointer
import jcuda.jcublas.JCublas2.cublasCreate
import jcuda.jcublas.JCublas2.cublasDestroy
import jcuda.jcublas.cublasHandle
import jcuda.runtime.JCuda.cudaFree
import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Test
import shape.komputation.cuda.allocateDeviceMemory
import shape.komputation.cuda.getVector
import shape.komputation.cuda.setVector
import shape.komputation.matrix.FloatMatrix
import shape.komputation.matrix.floatScalar

class CublasProjectionForwardTest {

    @Test
    fun testOneByOne() {

        val weights = floatScalar(2.0f)
        val input = floatScalar(3.0f)
        val expected = floatArrayOf(6.0f)

        check(weights, input, expected)
        check(input, weights, expected)

    }

    @Test
    fun testOneByOneWithBias() {

        val weights = floatScalar(2.0f)
        val input = floatScalar(3.0f)
        val bias = floatArrayOf(2.0f)

        val expected = floatArrayOf(8.0f)

        checkWithBias(weights, bias, input, expected)
        checkWithBias(input, bias, weights, expected)

    }

    @Test
    fun testOneByTwoTimesTwoByOne() {

        /*
                    3.0
                    4.0
            1.0 2.0 11.0
         */
        val weights = FloatMatrix(1, 2, floatArrayOf(1.0f, 2.0f))
        val input = FloatMatrix(2, 1, floatArrayOf(3.0f, 4.0f))

        check(weights, input, floatArrayOf(11.0f))

    }

    @Test
    fun testOneByTwoTimesTwoByOneWithBias() {

        val weights = FloatMatrix(1, 2, floatArrayOf(1.0f, 2.0f))
        val bias = floatArrayOf(5.0f)
        val input = FloatMatrix(2, 1, floatArrayOf(3.0f, 4.0f))

        checkWithBias(weights, bias, input, floatArrayOf(16.0f))

    }

    private fun check(weightMatrix : FloatMatrix, inputMatrix: FloatMatrix, expected : FloatArray) {

        val cublasHandle = cublasHandle()
        cublasCreate(cublasHandle)

        val deviceInput = Pointer()
        setVector(inputMatrix.entries, inputMatrix.entries.size, deviceInput)

        val deviceResult = Pointer()
        val resultDimension = weightMatrix.numberRows
        allocateDeviceMemory(deviceResult, resultDimension)

        val deviceWeights = Pointer()
        setVector(weightMatrix.entries, weightMatrix.entries.size, deviceWeights)

        cublasProject(
            cublasHandle,
            deviceInput,
            deviceWeights,
            weightMatrix.numberRows,
            weightMatrix.numberColumns,
            deviceResult)

        val actual = getVector(deviceResult, resultDimension)

        cudaFree(deviceInput)
        cudaFree(deviceResult)
        cudaFree(deviceWeights)

        cublasDestroy(cublasHandle)

        assertArrayEquals(expected, actual, 0.001f)

    }

    private fun checkWithBias(weightMatrix : FloatMatrix, bias : FloatArray, inputMatrix: FloatMatrix, expected : FloatArray) {

        val cublasHandle = cublasHandle()
        cublasCreate(cublasHandle)

        val deviceInput = Pointer()
        setVector(inputMatrix.entries, inputMatrix.entries.size, deviceInput)
        val deviceResult = Pointer()
        val resultDimension = weightMatrix.numberRows
        allocateDeviceMemory(deviceResult, resultDimension)

        val deviceWeights = Pointer()
        setVector(weightMatrix.entries, weightMatrix.entries.size, deviceWeights)
        val deviceBias = Pointer()
        setVector(bias, bias.size, deviceBias)

        cublasProjectWithBias(
            cublasHandle,
            deviceInput,
            deviceWeights,
            weightMatrix.numberRows,
            weightMatrix.numberColumns,
            deviceBias,
            bias.size,
            deviceResult)

        val actual = getVector(deviceResult, resultDimension)

        cudaFree(deviceInput)
        cudaFree(deviceResult)
        cudaFree(deviceWeights)
        cudaFree(deviceBias)

        cublasDestroy(cublasHandle)

        assertArrayEquals(expected, actual, 0.001f)

    }

}
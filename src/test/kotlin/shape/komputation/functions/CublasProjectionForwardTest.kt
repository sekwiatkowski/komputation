package shape.komputation.functions

import jcuda.Pointer
import jcuda.jcublas.JCublas2.cublasCreate
import jcuda.jcublas.JCublas2.cublasDestroy
import jcuda.jcublas.cublasHandle
import jcuda.runtime.JCuda.cudaFree
import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Test
import shape.komputation.cuda.allocateDeviceMemory
import shape.komputation.cuda.copyFromHostToDevice
import shape.komputation.matrix.DoubleMatrix
import shape.komputation.matrix.doubleScalar

class CublasProjectionForwardTest {

    @Test
    fun testOneByOne() {

        val weights = doubleScalar(2.0)
        val input = doubleScalar(3.0)
        val expected = doubleArrayOf(6.0)

        check(weights, input, expected)
        check(input, weights, expected)

    }

    @Test
    fun testOneByOneWithBias() {

        val weights = doubleScalar(2.0)
        val input = doubleScalar(3.0)
        val bias = doubleArrayOf(2.0)

        val expected = doubleArrayOf(8.0)

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
        val weights = DoubleMatrix(1, 2, doubleArrayOf(1.0, 2.0))
        val input = DoubleMatrix(2, 1, doubleArrayOf(3.0, 4.0))

        check(weights, input, doubleArrayOf(11.0))

    }

    @Test
    fun testOneByTwoTimesTwoByOneWithBias() {

        val weights = DoubleMatrix(1, 2, doubleArrayOf(1.0, 2.0))
        val bias = doubleArrayOf(5.0)
        val input = DoubleMatrix(2, 1, doubleArrayOf(3.0, 4.0))

        checkWithBias(weights, bias, input, doubleArrayOf(16.0))

    }

    private fun check(weightMatrix : DoubleMatrix, inputMatrix: DoubleMatrix, expected : DoubleArray) {

        val cublasHandle = cublasHandle()
        cublasCreate(cublasHandle)

        val deviceInput = copyFromHostToDevice(inputMatrix.entries, inputMatrix.entries.size)
        val deviceResult = Pointer()
        allocateDeviceMemory(deviceResult, weightMatrix.numberRows)
        val deviceWeights = copyFromHostToDevice(weightMatrix.entries, weightMatrix.entries.size)

        val actual = cublasProject(
            cublasHandle,
            deviceInput,
            deviceResult,
            deviceWeights,
            weightMatrix.numberRows,
            weightMatrix.numberColumns)

        cudaFree(deviceInput)
        cudaFree(deviceResult)
        cudaFree(deviceWeights)

        cublasDestroy(cublasHandle)

        assertArrayEquals(expected, actual, 0.001)

    }

    private fun checkWithBias(weightMatrix : DoubleMatrix, bias : DoubleArray, inputMatrix: DoubleMatrix, expected : DoubleArray) {

        val cublasHandle = cublasHandle()
        cublasCreate(cublasHandle)

        val deviceInput = copyFromHostToDevice(inputMatrix.entries, inputMatrix.entries.size)
        val deviceResult = Pointer()
        allocateDeviceMemory(deviceResult, weightMatrix.numberRows)
        val deviceWeights = copyFromHostToDevice(weightMatrix.entries, weightMatrix.entries.size)
        val deviceBias = copyFromHostToDevice(bias, bias.size)

        val actual = cublasProject(
            cublasHandle,
            deviceInput,
            deviceResult,
            deviceWeights,
            weightMatrix.numberRows,
            weightMatrix.numberColumns,
            deviceBias,
            bias.size)

        cudaFree(deviceInput)
        cudaFree(deviceResult)
        cudaFree(deviceWeights)
        cudaFree(deviceBias)

        cublasDestroy(cublasHandle)

        assertArrayEquals(expected, actual, 0.001)

    }

}
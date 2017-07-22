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
import shape.komputation.matrix.floatRowVector
import shape.komputation.matrix.floatScalar

class CublasProjectionBackwardTest {

    @Test
    fun testBackwardProjectionWrtInput1() {

        val weights = floatArrayOf(2.0f)
        val chain = floatArrayOf(3.0f)
        val expected = floatArrayOf(6.0f)

        checkBackwardProjectionWrtInput(1, 1, weights, chain, expected)

    }

    @Test
    fun testBackwardProjectionWrtInput2() {

        /*
            weights = 2
                      3
            chain = 4
                    5

                             4
                             5
            weights^T >> 2 3
         */

        val weights = floatArrayOf(2.0f, 3.0f)
        val chain = floatArrayOf(4.0f, 5.0f)
        val expected = floatArrayOf(2.0f*4.0f + 3.0f*5.0f)

        checkBackwardProjectionWrtInput(2, 1, weights, chain, expected)

    }

    @Test
    fun testBackwardProjectionWrtInput3() {

        /*
            weights = 2 3
                      4 5
            chain = 6
                    7

                              6
                              7
            weights^T >> 2 4 40
                         3 5 53
         */

        val weights = floatArrayOf(2.0f, 4.0f, 3.0f, 5.0f)
        val chain = floatArrayOf(6.0f, 7.0f)
        val expected = floatArrayOf(40.0f, 53.0f)

        checkBackwardProjectionWrtInput(2, 2, weights, chain, expected)

    }

    /*
    gemv solution:
            x1 x2 x3 << transposed x
    chain_1
    chain_2
    */
    @Test
    fun testBackwardProjectionWrtWeight1() {

        val weights = floatScalar(2.0f)
        val chain = floatArrayOf(3.0f)
        val expected = floatArrayOf(6.0f)

        checkBackwardProjectionWrtWeights(weights, chain, expected)

    }

    /*
           2  3
        4  8 12
        5 10 15
     */

    @Test
    fun testBackwardProjectionWrtWeight2() {

        val weights = floatRowVector(2.0f, 3.0f)
        val chain = floatArrayOf(4.0f, 5.0f)
        val expected = floatArrayOf(8.0f, 10.0f, 12.0f, 15.0f)

        checkBackwardProjectionWrtWeights(weights, chain, expected)

    }

    private fun checkBackwardProjectionWrtInput(numberWeightRows : Int, numberWeightColumns : Int, weights: FloatArray, chain: FloatArray, expected: FloatArray) {

        val numberWeightEntries = numberWeightRows * numberWeightColumns

        val cublasHandle = cublasHandle()
        cublasCreate(cublasHandle)

        val deviceWeights = Pointer()
        setVector(weights, numberWeightEntries, deviceWeights)

        val deviceChain = Pointer()
        setVector(chain, numberWeightRows, deviceChain)

        val deviceResult = Pointer()
        allocateDeviceMemory(deviceResult, numberWeightColumns)

        cublasBackwardProjectionWrtInput(cublasHandle, deviceWeights, numberWeightRows, numberWeightColumns, deviceChain, deviceResult)

        val hostResult = getVector(deviceResult, numberWeightColumns)

        cudaFree(deviceWeights)
        cudaFree(deviceChain)
        cudaFree(deviceResult)

        cublasDestroy(cublasHandle)

        assertArrayEquals(expected, hostResult, 0.001f)

    }

    private fun checkBackwardProjectionWrtWeights(input : FloatMatrix, chain : FloatArray, expected: FloatArray) {

        val inputEntries = input.entries
        val inputDimension = inputEntries.size

        val chainDimension = chain.size

        val resultDimension = inputDimension * chainDimension

        val cublasHandle = cublasHandle()
        cublasCreate(cublasHandle)

        val deviceInput = Pointer()
        setVector(inputEntries, inputDimension, deviceInput)

        val deviceChain = Pointer()
        setVector(chain, chainDimension, deviceChain)

        val deviceResult = Pointer()
        allocateDeviceMemory(deviceResult, resultDimension)

        cublasBackwardProjectionWrtWeights(cublasHandle, deviceInput, deviceChain, deviceResult, chainDimension, inputDimension)

        val hostResult = getVector(deviceResult, resultDimension)

        cudaFree(deviceInput)
        cudaFree(deviceChain)
        cudaFree(deviceResult)

        cublasDestroy(cublasHandle)

        assertArrayEquals(expected, hostResult, 0.001f)

    }


}
package shape.komputation.cuda.functions

import jcuda.Pointer
import jcuda.Sizeof
import jcuda.jcublas.JCublas2.*
import jcuda.jcublas.cublasHandle
import jcuda.runtime.JCuda.cudaFree
import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Test
import shape.komputation.cuda.copyFromHostToDevice
import shape.komputation.matrix.DoubleMatrix
import shape.komputation.matrix.doubleColumnVector
import shape.komputation.matrix.doubleRowVector
import shape.komputation.matrix.doubleScalar

class CublasProjectionBackwardTest {

    @Test
    fun testBackwardProjectionWrtInput1() {

        val weights = doubleScalar(2.0)
        val chain = doubleArrayOf(3.0)
        val expected = doubleArrayOf(6.0)

        checkBackwardProjectionWrtInput(weights, chain, expected)

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

        val weights = doubleColumnVector(2.0, 3.0)
        val chain = doubleArrayOf(4.0, 5.0)
        val expected = doubleArrayOf(2.0*4.0 + 3.0*5.0)

        checkBackwardProjectionWrtInput(weights, chain, expected)

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

        val weights = DoubleMatrix(2, 2, doubleArrayOf(2.0, 4.0, 3.0, 5.0))
        val chain = doubleArrayOf(6.0, 7.0)
        val expected = doubleArrayOf(40.0, 53.0)

        checkBackwardProjectionWrtInput(weights, chain, expected)

    }

    /*
    gemv solution:
            x1 x2 x3 << transposed x
    chain_1
    chain_2
    */
    @Test
    fun testBackwardProjectionWrtWeight1() {

        val weights = doubleScalar(2.0)
        val chain = doubleArrayOf(3.0)
        val expected = doubleArrayOf(6.0)

        checkBackwardProjectionWrtWeights(weights, chain, expected)

    }

    /*
           2  3
        4  8 12
        5 10 15
     */

    @Test
    fun testBackwardProjectionWrtWeight2() {

        val weights = doubleRowVector(2.0, 3.0)
        val chain = doubleArrayOf(4.0, 5.0)
        val expected = doubleArrayOf(8.0, 10.0, 12.0, 15.0)

        checkBackwardProjectionWrtWeights(weights, chain, expected)

    }

    private fun checkBackwardProjectionWrtInput(weights: DoubleMatrix, chain: DoubleArray, expected: DoubleArray) {

        val numberWeightRows = weights.numberRows
        val numberWeightColumns = weights.numberColumns
        val numberWeightEntries = numberWeightRows * numberWeightColumns

        val cublasHandle = cublasHandle()
        cublasCreate(cublasHandle)

        val deviceWeights = Pointer()
        copyFromHostToDevice(weights.entries, numberWeightEntries, deviceWeights)

        val deviceChain = Pointer()
        copyFromHostToDevice(chain, numberWeightRows, deviceChain)

        val hostResult = DoubleArray(numberWeightColumns)
        val deviceResult = Pointer()
        copyFromHostToDevice(hostResult, numberWeightColumns, deviceResult)

        cublasBackwardProjectionWrtInput(cublasHandle, deviceWeights, numberWeightRows, numberWeightColumns, deviceChain, deviceResult)

        cublasGetVector(numberWeightColumns, Sizeof.DOUBLE, deviceResult, 1, Pointer.to(hostResult), 1)

        cudaFree(deviceWeights)
        cudaFree(deviceChain)
        cudaFree(deviceResult)

        cublasDestroy(cublasHandle)

        assertArrayEquals(expected, hostResult, 0.001)

    }

    private fun checkBackwardProjectionWrtWeights(input : DoubleMatrix, chain : DoubleArray, expected: DoubleArray) {

        val inputEntries = input.entries
        val inputDimension = inputEntries.size

        val chainDimension = chain.size

        val resultDimension = inputDimension * chainDimension

        val cublasHandle = cublasHandle()
        cublasCreate(cublasHandle)

        val deviceInput = Pointer()
        copyFromHostToDevice(inputEntries, inputDimension, deviceInput)

        val deviceChain = Pointer()
        copyFromHostToDevice(chain, chainDimension, deviceChain)

        val hostResult = DoubleArray(resultDimension)
        val deviceResult = Pointer()
        copyFromHostToDevice(hostResult, resultDimension, deviceResult)

        cublasBackwardProjectionWrtWeights(cublasHandle, deviceInput, deviceChain, deviceResult, chainDimension, inputDimension)

        cublasGetVector(resultDimension, Sizeof.DOUBLE, deviceResult, 1, Pointer.to(hostResult), 1)

        cudaFree(deviceInput)
        cudaFree(deviceChain)
        cudaFree(deviceResult)

        cublasDestroy(cublasHandle)

        assertArrayEquals(expected, hostResult, 0.001)

    }


}
package shape.komputation.functions

import jcuda.Pointer
import jcuda.Sizeof
import jcuda.jcublas.JCublas2.*
import jcuda.jcublas.cublasHandle
import jcuda.runtime.JCuda.cudaFree
import jcuda.runtime.JCuda.cudaMalloc
import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Test
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
        cudaMalloc(deviceWeights, (numberWeightEntries * Sizeof.DOUBLE).toLong())
        cublasSetVector(numberWeightEntries, Sizeof.DOUBLE, Pointer.to(weights.entries), 1, deviceWeights, 1)

        val deviceChain = Pointer()
        cudaMalloc(deviceChain, (chain.size * Sizeof.DOUBLE).toLong())
        cublasSetVector(numberWeightRows, Sizeof.DOUBLE, Pointer.to(chain), 1, deviceChain, 1)

        val deviceResult = Pointer()
        cudaMalloc(deviceResult, (numberWeightColumns * Sizeof.DOUBLE).toLong())
        val hostResult = DoubleArray(numberWeightColumns)
        cublasSetVector(numberWeightColumns, Sizeof.DOUBLE, Pointer.to(hostResult), 1, deviceResult, 1)

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

        val cublasHandle = cublasHandle()
        cublasCreate(cublasHandle)

        val deviceInput = Pointer()
        cudaMalloc(deviceInput, (inputDimension * Sizeof.DOUBLE).toLong())
        cublasSetVector(inputDimension, Sizeof.DOUBLE, Pointer.to(inputEntries), 1, deviceInput, 1)

        val deviceChain = Pointer()
        cudaMalloc(deviceChain, (chainDimension * Sizeof.DOUBLE).toLong())
        cublasSetVector(chainDimension, Sizeof.DOUBLE, Pointer.to(chain), 1, deviceChain, 1)

        val resultDimension = inputDimension * chainDimension

        val deviceResult = Pointer()
        cudaMalloc(deviceResult, (resultDimension * Sizeof.DOUBLE).toLong())
        val hostResult = DoubleArray(resultDimension)
        cublasSetVector(resultDimension, Sizeof.DOUBLE, Pointer.to(hostResult), 1, deviceResult, 1)

        cublasBackwardProjectionWrtWeights(cublasHandle, deviceInput, deviceChain, chainDimension, deviceResult)

        cublasGetVector(resultDimension, Sizeof.DOUBLE, deviceResult, 1, Pointer.to(hostResult), 1)

        cudaFree(deviceInput)
        cudaFree(deviceChain)
        cudaFree(deviceResult)

        cublasDestroy(cublasHandle)

        assertArrayEquals(expected, hostResult, 0.001)

    }


}
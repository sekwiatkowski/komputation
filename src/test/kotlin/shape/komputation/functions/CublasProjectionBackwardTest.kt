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

        val numberWeightEntries = weights.numberRows * weights.numberColumns

        val cublasHandle = cublasHandle()
        cublasCreate(cublasHandle)

        val deviceWeights = Pointer()
        val deviceChain = Pointer()

        cudaMalloc(deviceWeights, (numberWeightEntries * Sizeof.DOUBLE).toLong())
        cudaMalloc(deviceChain, (chain.size * Sizeof.DOUBLE).toLong())

        cublasSetVector(numberWeightEntries, Sizeof.DOUBLE, Pointer.to(weights.entries), 1, deviceWeights, 1)
        cublasSetVector(weights.numberRows, Sizeof.DOUBLE, Pointer.to(chain), 1, deviceChain, 1)

        val actual = cublasBackwardProjectionWrtInput(cublasHandle, deviceWeights, weights.numberRows, weights.numberColumns, deviceChain)

        cudaFree(deviceWeights)
        cudaFree(deviceChain)

        cublasDestroy(cublasHandle)

        assertArrayEquals(expected, actual, 0.001)

    }

    private fun checkBackwardProjectionWrtWeights(input : DoubleMatrix, chain : DoubleArray, expected: DoubleArray) {

        val inputEntries = input.entries
        val inputDimension = inputEntries.size

        val chainDimension = chain.size

        val cublasHandle = cublasHandle()
        cublasCreate(cublasHandle)

        val deviceInput = Pointer()
        val deviceChain = Pointer()

        cudaMalloc(deviceInput, (inputDimension * Sizeof.DOUBLE).toLong())
        cudaMalloc(deviceChain, (chainDimension * Sizeof.DOUBLE).toLong())

        cublasSetVector(chainDimension, Sizeof.DOUBLE, Pointer.to(chain), 1, deviceChain, 1)
        cublasSetVector(inputDimension, Sizeof.DOUBLE, Pointer.to(inputEntries), 1, deviceInput, 1)

        val actual = cublasBackwardProjectionWrtWeights(cublasHandle, deviceInput, deviceChain, chainDimension, inputDimension * chainDimension)

        cudaFree(deviceInput)
        cudaFree(deviceChain)

        cublasDestroy(cublasHandle)

        assertArrayEquals(expected, actual, 0.001)

    }


}
package shape.komputation.functions

import jcuda.Pointer
import jcuda.Sizeof
import jcuda.jcublas.JCublas.*
import shape.komputation.matrix.createBlasMatrix
import shape.komputation.matrix.createBlasVector

fun project(input: DoubleArray, numberInputRows : Int, numberInputColumns: Int, weights: DoubleArray, numberWeightRows : Int, numberWeightColumns : Int) =

    project(input, numberInputRows, numberInputColumns, weights, numberWeightRows, numberWeightColumns, null)

fun project(input: DoubleArray, numberInputRows : Int, numberInputColumns: Int, weights: DoubleArray, numberWeightRows : Int, numberWeightColumns : Int, bias : DoubleArray?) : DoubleArray {

    val inputMatrix = createBlasMatrix(numberInputRows, numberInputColumns, input)
    val weightMatrix = createBlasMatrix(numberWeightRows, numberWeightColumns, weights)

    if (bias != null) {

        if (numberInputColumns == 1) {

            val biasVector = createBlasVector(bias, true)

            return weightMatrix.multiplyAdd(inputMatrix, biasVector).getEntries()

        }
        else {

            val expandedBias = repeatColumn(bias, numberInputColumns)

            return weightMatrix.multiplyAdd(inputMatrix, createBlasMatrix(bias.size, numberInputColumns, expandedBias)).getEntries()

        }

    }
    else {

        return weightMatrix.multiply(inputMatrix).getEntries()

    }

}

fun backwardProjectionWrtInput(
    numberInputRows: Int,
    numberInputColumns : Int,
    numberInputEntries : Int,
    weightEntries : DoubleArray,
    numberWeightRows : Int,
    chainEntries: DoubleArray,
    numberChainRows : Int): DoubleArray {

    val derivatives = DoubleArray(numberInputEntries)

    var index = 0

    for (indexInputColumn in 0..numberInputColumns - 1) {

        for (indexInputRow in 0..numberInputRows - 1) {

            var derivative = 0.0

            for (indexWeightRow in 0..numberWeightRows - 1) {

                val chainEntry = chainEntries[indexWeightRow + indexInputColumn * numberChainRows]
                val weightEntry = weightEntries[indexWeightRow + indexInputRow * numberWeightRows]

                derivative += chainEntry * weightEntry

            }

            derivatives[index++] = derivative

        }

    }

    return derivatives

}

/*
        Differentiation w.r.t input:

        d Wx / d x = w_11 + w_21
                     w_12 + w_22
                     w_13 + w_23

        gemv solution:
                                  chain_1
                                  chain_2
        transposed W >> w_11 w_21
                        w_12 w_22
                        w_13 w_23
 */
fun cublasBackwardProjectionWrtInput(hostWeights: DoubleArray, numberWeightRows: Int, numberWeightColumns: Int, numberWeightEntries: Int, hostChain: DoubleArray): DoubleArray {

    cublasInit()

    val deviceWeights = Pointer()
    val deviceChain = Pointer()
    val deviceResult = Pointer()

    cublasAlloc(numberWeightEntries, Sizeof.DOUBLE, deviceWeights)
    cublasAlloc(numberWeightRows, Sizeof.DOUBLE, deviceChain)
    cublasAlloc(numberWeightColumns, Sizeof.DOUBLE, deviceResult)

    val hostResult = DoubleArray(numberWeightColumns)

    cublasSetVector(numberWeightEntries, Sizeof.DOUBLE, Pointer.to(hostWeights), 1, deviceWeights, 1)
    cublasSetVector(numberWeightRows, Sizeof.DOUBLE, Pointer.to(hostChain), 1, deviceChain, 1)
    cublasSetVector(numberWeightColumns, Sizeof.DOUBLE, Pointer.to(hostResult), 1, deviceResult, 1)

    cublasDgemv(
        't', // transpose
        numberWeightRows, // number of rows of matrix A
        numberWeightColumns, // number of columns of matrix A
        1.0, // alpha
        deviceWeights, // weight pointer
        numberWeightRows, // number weight rows
        deviceChain, // input pointer
        1, // storage spacing between elements of x
        0.0, // beta
        deviceResult, // result pointer
        1 // specifies the storage spacing between elements of y
    )

    cublasGetVector(numberWeightColumns, Sizeof.DOUBLE, deviceResult, 1, Pointer.to(hostResult), 1)

    cublasFree(deviceWeights)
    cublasFree(deviceChain)
    cublasFree(deviceResult)

    cublasShutdown()

    return hostResult

}

fun backwardProjectionWrtWeights(
    numberWeightEntries : Int,
    numberWeightRows : Int,
    numberWeightColumns: Int,
    inputEntries: DoubleArray,
    numberInputRows : Int,
    chainEntries: DoubleArray,
    numberChainRows: Int,
    numberChainColumns : Int): DoubleArray {

    val derivatives = DoubleArray(numberWeightEntries)

    var index = 0

    for (indexWeightColumn in 0..numberWeightColumns - 1) {

        for (indexWeightRow in 0..numberWeightRows - 1) {

            var derivative = 0.0

            for (indexChainColumn in 0..numberChainColumns - 1) {

                // d loss / d pre1, d loss / d pre2
                // All multiplications on other rows equal to zero
                val chainEntry = chainEntries[indexWeightRow + indexChainColumn * numberChainRows]

                // d pre ij / d wk
                val inputEntry = inputEntries[indexWeightColumn + indexChainColumn * numberInputRows]

                derivative += chainEntry * inputEntry

            }

            derivatives[index++] = derivative

        }
    }

    return derivatives

}

fun backwardProjectionWrtBias(numberBiasRows : Int, chain: DoubleArray, numberChainRows: Int, numberChainColumns: Int) =

    DoubleArray(numberBiasRows) { indexRow ->

        var derivative = 0.0

        for (indexChainColumn in 0..numberChainColumns - 1) {

            derivative += chain[numberChainRows * indexChainColumn + indexRow]

        }

        derivative

    }

package shape.komputation.cpu.functions

import java.util.*

fun computeNumberFilterRowPositions(numberRows : Int, filterHeight: Int) =

    numberRows - filterHeight + 1

fun computeNumberFilterColumnPositions(numberColumns : Int, filterWidth: Int) =

    numberColumns - filterWidth + 1

fun expandForConvolution(
    inputEntries : FloatArray,
    numberInputRows: Int,
    resultEntries : FloatArray,
    numberResultEntries : Int,
    numberFilterRowPositions: Int,
    numberFilterColumnPositions: Int,
    filterWidth : Int,
    filterHeight: Int) {

    var count = 0

    for (columnStart in 0..numberFilterColumnPositions - 1) {

        for (rowStart in 0..numberFilterRowPositions - 1) {

            for (indexColumn in columnStart..columnStart + filterWidth - 1) {

                for (indexRow in rowStart..rowStart + filterHeight - 1) {

                    resultEntries[count++] = inputEntries[indexRow + indexColumn * numberInputRows]

                }

            }

        }

    }

    Arrays.fill(resultEntries, count, numberResultEntries, Float.NaN)

}

fun backwardExpansionForConvolution(
    numberInputRows: Int,
    numberActualEntries: Int,
    numberResultEntries: Int,
    resultEntries: FloatArray,
    filterHeight: Int,
    numberFilterRowPositions: Int,
    numberFilterColumnPositions: Int,
    chain: FloatArray,
    numberChainRows: Int) {

    var count = 0

    Arrays.fill(resultEntries, 0, numberActualEntries, 0f)

    for (indexConvolution in 0..numberFilterColumnPositions - 1) {

        val firstColumnOfConvolution = firstColumnOfConvolution(indexConvolution, numberFilterRowPositions)
        val firstRowOfConvolution = firstRowOfConvolution(indexConvolution, numberFilterRowPositions)

        for (indexConvolutionEntry in 0..numberChainRows - 1) {

            val columnInConvolution = columnInConvolution(indexConvolutionEntry, filterHeight)
            val column = firstColumnOfConvolution + columnInConvolution

            val rowInConvolution = rowInConvolution(indexConvolutionEntry, filterHeight)
            val row = firstRowOfConvolution + rowInConvolution

            val derivative = chain[count++]
            resultEntries[row + column * numberInputRows] += derivative

        }

    }

    Arrays.fill(resultEntries, numberActualEntries, numberResultEntries, Float.NaN)

}

fun firstColumnOfConvolution(indexConvolution: Int, numberFilterRowPositions: Int) =

    indexConvolution / numberFilterRowPositions

fun firstRowOfConvolution(indexConvolution: Int, numberFilterRowPositions: Int) =

    indexConvolution % numberFilterRowPositions

fun columnInConvolution(indexConvolutionEntry : Int, filterHeight: Int) =

    indexConvolutionEntry / filterHeight

fun rowInConvolution(indexConvolutionEntry: Int, filterHeight: Int) =

    indexConvolutionEntry % filterHeight
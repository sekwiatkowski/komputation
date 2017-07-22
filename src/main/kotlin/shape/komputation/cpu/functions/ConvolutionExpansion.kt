package shape.komputation.cpu.functions

import shape.komputation.matrix.FloatMatrix

fun convolutionsPerColumn(numberRows : Int, filterHeight: Int) =

    numberRows - filterHeight + 1

fun convolutionsPerRow(numberColumns : Int, filterWidth: Int) =

    numberColumns - filterWidth + 1

fun expandForConvolution(
    inputEntries : FloatArray,
    numberInputRows : Int,
    convolutionsPerRow : Int,
    convolutionsPerColumn : Int,
    filterWidth : Int,
    filterHeight: Int): FloatMatrix {

    val numberConvolutions = convolutionsPerRow * convolutionsPerColumn

    val numberExpansionRows = filterWidth * filterHeight
    val numberExpansionColumns = numberConvolutions

    val expandedInputMatrix = FloatArray(numberExpansionRows * numberConvolutions)

    var counter = 0

    for (indexConvolutionStartRow in 0..convolutionsPerColumn - 1) {

        for (indexConvolutionStartColumn in 0..convolutionsPerRow - 1) {

            for (indexColumn in indexConvolutionStartColumn..indexConvolutionStartColumn + filterWidth - 1) {

                for (indexRow in indexConvolutionStartRow..indexConvolutionStartRow + filterHeight - 1) {

                    expandedInputMatrix[counter++] = inputEntries[indexRow + indexColumn * numberInputRows]

                }

            }

        }

    }

    return FloatMatrix(numberExpansionRows, numberExpansionColumns, expandedInputMatrix)

}

fun backwardExpansionForConvolution(
    numberInputRows: Int,
    numberInputColumns: Int,
    filterHeight: Int,
    convolutionsPerRow: Int,
    chain: FloatArray,
    numberChainRows: Int,
    numberChainColumns: Int): FloatArray {

    val sums = FloatArray(numberInputRows * numberInputColumns)

    var count = 0

    for (indexConvolution in 0..numberChainColumns - 1) {

        val firstColumnOfConvolution = firstColumnOfConvolution(indexConvolution, convolutionsPerRow)
        val firstRowOfConvolution = firstRowOfConvolution(indexConvolution, convolutionsPerRow)

        for (indexConvolutionEntry in 0..numberChainRows - 1) {

            val columnInConvolution = columnInConvolution(indexConvolutionEntry, filterHeight)
            val column = firstColumnOfConvolution + columnInConvolution

            val rowInConvolution = rowInConvolution(indexConvolutionEntry, filterHeight)
            val row = firstRowOfConvolution + rowInConvolution

            val derivative = chain[count++]
            sums[row + column * numberInputRows] += derivative

        }

    }

    return sums

}

fun firstColumnOfConvolution(indexConvolution: Int, convolutionsPerRow: Int) =

    indexConvolution % convolutionsPerRow

fun firstRowOfConvolution(indexConvolution: Int, convolutionsPerColumn: Int) =

    indexConvolution / convolutionsPerColumn

fun columnInConvolution(indexConvolutionEntry : Int, filterHeight: Int) =

    indexConvolutionEntry / filterHeight

fun rowInConvolution(indexConvolutionEntry: Int, filterHeight: Int) =

    indexConvolutionEntry % filterHeight
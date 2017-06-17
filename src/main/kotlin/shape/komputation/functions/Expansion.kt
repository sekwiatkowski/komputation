package shape.komputation.functions

import shape.komputation.matrix.DoubleMatrix

fun convolutionsPerColumn(numberRows : Int, filterHeight: Int) =

    numberRows - filterHeight + 1

fun convolutionsPerRow(numberColumns : Int, filterWidth: Int) =

    numberColumns - filterWidth + 1

fun expand(inputEntries : DoubleArray, numberInputRows : Int, numberInputColumns : Int, filterWidth : Int, filterHeight: Int): DoubleMatrix {

    val convolutionsPerRow = convolutionsPerRow(numberInputColumns, filterWidth)
    val convolutionsPerColumn = convolutionsPerColumn(numberInputRows, filterHeight)

    val numberConvolutions = convolutionsPerRow * convolutionsPerColumn

    val numberExpansionRows = filterWidth * filterHeight
    val numberExpansionColumns = numberConvolutions

    val expandedInputMatrix = DoubleArray(numberExpansionRows * numberConvolutions)

    var counter = 0

    for (indexConvolutionStartRow in 0..convolutionsPerColumn - 1) {

        for (indexConvolutionStartColumn in 0..convolutionsPerRow - 1) {

            for (indexRow in indexConvolutionStartRow..indexConvolutionStartRow + filterHeight - 1) {

                for (indexColumn in indexConvolutionStartColumn..indexConvolutionStartColumn + filterWidth - 1) {

                    expandedInputMatrix[counter++] = inputEntries[indexRow + indexColumn * numberInputRows]

                }

            }

        }

    }

    return DoubleMatrix(numberExpansionRows, numberExpansionColumns, expandedInputMatrix)

}


fun backwardExpansion(
    numberInputRows: Int,
    numberInputColumns: Int,
    filterWidth: Int,
    chain: DoubleArray,
    numberChainRows: Int,
    numberChainColumns: Int): DoubleArray {

    val sums = DoubleArray(numberInputRows * numberInputColumns)

    val convolutionsPerRow = convolutionsPerRow(numberInputColumns, filterWidth)

    var count = 0

    for (indexConvolution in 0..numberChainColumns - 1) {

        for (indexConvolutionEntry in 0..numberChainRows - 1) {

            val derivative = chain[count++]

            val inputRow = expandedRowToOriginalRow(indexConvolution, indexConvolutionEntry, convolutionsPerRow, filterWidth)
            val inputColumn = expandedColumnToOriginalColumn(indexConvolution, indexConvolutionEntry, convolutionsPerRow, filterWidth)

            sums[inputColumn * numberInputRows + inputRow] += derivative

        }

    }

    return sums

}

fun expandedRowToOriginalRow(indexConvolution : Int, indexConvolutionEntry : Int, perRow : Int, filterWidth : Int) =

    indexConvolution / perRow + indexConvolutionEntry / filterWidth

fun expandedColumnToOriginalColumn(indexConvolution: Int, indexConvolutionEntry: Int, perRow: Int, filterWidth: Int) =

    indexConvolution % perRow + indexConvolutionEntry % filterWidth
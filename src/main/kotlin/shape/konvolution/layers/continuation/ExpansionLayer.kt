package shape.konvolution.layers.continuation

import shape.konvolution.BackwardResult
import shape.konvolution.matrix.RealMatrix
import shape.konvolution.matrix.createRealMatrix

class ExpansionLayer(private val filterWidth: Int, private val filterHeight: Int) : ContinuationLayer {

    override fun forward(input: RealMatrix) =

        expandMatrixForConvolution(input, filterWidth, filterHeight)

    override fun backward(input: RealMatrix, output: RealMatrix, chain: RealMatrix): BackwardResult {

        val collectedGradients = collectGradients(input.numberRows(), input.numberColumns(), filterWidth, chain)

        return BackwardResult(collectedGradients)

    }

}

fun convolutionsPerColumn(numberRows : Int, filterHeight: Int) =

    numberRows - filterHeight + 1

fun convolutionsPerRow(numberColumns : Int, filterWidth: Int) =

    numberColumns - filterWidth + 1

fun expandMatrixForConvolution(input: RealMatrix, filterWidth : Int, filterHeight: Int): RealMatrix {

    val convolutionsPerRow = convolutionsPerRow(input.numberColumns(), filterWidth)
    val convolutionsPerColumn = convolutionsPerColumn(input.numberRows(), filterHeight)

    val numberConvolutions = convolutionsPerRow * convolutionsPerColumn

    val expandedInputMatrix = createRealMatrix(filterWidth * filterHeight, numberConvolutions)

    var indexConvolution = 0

    for (startRow in 0..convolutionsPerColumn - 1) {

        for (startColumn in 0..convolutionsPerRow - 1) {

            var indexConvolutionEntry = 0

            for (indexRow in startRow..startRow + filterHeight - 1) {

                for (indexColumn in startColumn..startColumn + filterWidth - 1) {

                    expandedInputMatrix.set(indexConvolutionEntry, indexConvolution, input.get(indexRow, indexColumn))

                    indexConvolutionEntry++

                }

            }

            indexConvolution++

        }

    }

    return expandedInputMatrix

}

fun collectGradients(
    inputRows : Int,
    inputColumns : Int,
    filterWidth: Int,
    chain: RealMatrix): RealMatrix {

    val inputGradient = createRealMatrix(inputRows, inputColumns)

    val convolutionsPerRow = convolutionsPerRow(inputColumns, filterWidth)

    for (indexConvolution in 0..chain.numberColumns() - 1) {

        for (indexConvolutionEntry in 0..chain.numberRows() - 1) {

            val inputRow = expandedRowToOriginalRow(indexConvolution, indexConvolutionEntry, convolutionsPerRow, filterWidth)
            val inputColumn = expandedColumnToOriginalColumn(indexConvolution, indexConvolutionEntry, convolutionsPerRow, filterWidth)

            val derivative = chain.get(indexConvolutionEntry, indexConvolution)

            inputGradient.add(inputRow, inputColumn, derivative)

        }

    }

    return inputGradient
}

fun expandedRowToOriginalRow(indexConvolution : Int, indexConvolutionEntry : Int, perRow : Int, filterWidth : Int) =

    indexConvolution / perRow + indexConvolutionEntry / filterWidth

fun expandedColumnToOriginalColumn(indexConvolution: Int, indexConvolutionEntry: Int, perRow: Int, filterWidth: Int) =

    indexConvolution % perRow + indexConvolutionEntry % filterWidth
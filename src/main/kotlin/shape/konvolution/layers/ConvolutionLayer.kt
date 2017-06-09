package shape.konvolution.layers

import no.uib.cipr.matrix.DenseMatrix
import no.uib.cipr.matrix.Matrix
import shape.konvolution.*
import shape.konvolution.optimization.Optimizable
import shape.konvolution.optimization.Optimizer

class ConvolutionLayer(
    private val filterWidth: Int,
    private val filterHeight: Int,
    private val weights : Matrix,
    private val bias : Matrix? = null,
    private val weightOptimizer: Optimizer? = null,
    private val biasOptimizer: Optimizer? = null) : Layer, Optimizable {

    private val optimizeWeights = weightOptimizer != null
    private val optimizeBias = biasOptimizer != null

    override fun forward(input: Matrix) : Matrix {

        val expandedInputMatrix = expandMatrixForConvolution(input, filterWidth, filterHeight)

        val convolutions = project(expandedInputMatrix, weights, bias)

        return convolutions

    }

    override fun backward(input: Matrix, output : Matrix, chain : Matrix): BackwardResult {

        // # filters * # convolutions
        val expandedInputGradient = differentiateProjectionWrtInput(input, weights, chain)

        val convolutionsPerRow = convolutionsPerRow(input.numColumns(), filterWidth)

        val inputGradient = collectGradients(input.numRows(), input.numColumns(), expandedInputGradient, convolutionsPerRow)

        val parameterGradients = differentiateProjectionWrtParameters(input, weights, optimizeWeights, bias, optimizeBias, chain)

        return BackwardResult(inputGradient, parameterGradients)

    }

    fun collectGradients(inputRows : Int, inputColumns : Int, expandedInputGradient: DenseMatrix, convolutionsPerRow: Int): DenseMatrix {

        val inputGradient = createDenseMatrix(inputRows, inputColumns)

        for (indexConvolution in 0..expandedInputGradient.numColumns() - 1) {

            for (indexConvolutionEntry in 0..expandedInputGradient.numRows() - 1) {

                val inputRow = expandedRowToOriginalRow(indexConvolution, indexConvolutionEntry, convolutionsPerRow, filterWidth)
                val inputColumn = expandedColumnToOriginalColumn(indexConvolution, indexConvolutionEntry, convolutionsPerRow, filterWidth)

                val derivative = expandedInputGradient.get(indexConvolutionEntry, indexConvolution)

                inputGradient.add(inputRow, inputColumn, derivative)

            }

        }

        return inputGradient
    }

    override fun optimize(gradients: Array<Matrix?>) {

        if (this.weightOptimizer != null) {

            val weightGradient = gradients.first()

            this.weightOptimizer.optimize(this.weights, weightGradient!!)

        }

        if (this.biasOptimizer != null) {

            val biasGradient = gradients.last()

            this.biasOptimizer.optimize(this.bias!!, biasGradient!!)

        }

    }

}

fun createConvolutionLayer(
    numberFilters: Int,
    filterWidth: Int,
    filterHeight : Int,
    initializationStrategy : () -> Double,
    weightOptimizer: Optimizer,
    biasOptimizer: Optimizer): ConvolutionLayer {

    val weights = initializeMatrix(initializationStrategy, numberFilters,filterWidth * filterHeight)
    val bias = initializeBias(initializationStrategy, numberFilters)

    return ConvolutionLayer(filterWidth, filterHeight, weights, bias, weightOptimizer, biasOptimizer)

}

fun expandMatrixForConvolution(input: Matrix, filterWidth : Int, filterHeight: Int): Matrix {

    val convolutionsPerRow = input.numColumns() - filterWidth + 1
    val convolutionsPerColumn = input.numRows() - filterHeight + 1

    val numberConvolutions = convolutionsPerRow * convolutionsPerColumn

    val expandedInputMatrix = createDenseMatrix(filterWidth * filterHeight, numberConvolutions)

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

fun convolutionsPerRow(numberColumns : Int, filterWidth: Int) =

    numberColumns - filterWidth + 1

fun convolutionsPerColumn(numberRows : Int, filterHeight: Int) =

    numberRows - filterHeight + 1

fun expandedRowToOriginalRow(indexConvolution : Int, indexConvolutionEntry : Int, perRow : Int, filterWidth : Int) =

    indexConvolution / perRow + indexConvolutionEntry / filterWidth

fun expandedColumnToOriginalColumn(indexConvolution: Int, indexConvolutionEntry: Int, perRow: Int, filterWidth: Int) =

    indexConvolution % perRow + indexConvolutionEntry % filterWidth
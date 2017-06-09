package shape.konvolution.layers.continuation

import shape.konvolution.*
import shape.konvolution.optimization.Optimizable
import shape.konvolution.optimization.Optimizer

class ConvolutionLayer(
    private val filterWidth: Int,
    private val filterHeight: Int,
    private val weights : RealMatrix,
    private val bias : RealMatrix? = null,
    private val weightOptimizer: Optimizer? = null,
    private val biasOptimizer: Optimizer? = null) : ContinuationLayer, Optimizable {

    private val optimizeWeights = weightOptimizer != null
    private val optimizeBias = biasOptimizer != null

    override fun forward(input: RealMatrix) : RealMatrix {

        val expandedInputMatrix = expandMatrixForConvolution(input, filterWidth, filterHeight)

        val convolutions = project(expandedInputMatrix, weights, bias)

        return convolutions

    }

    override fun backward(input: RealMatrix, output : RealMatrix, chain : RealMatrix): BackwardResult {

        // # filters * # convolutions
        val expandedInputGradient = differentiateProjectionWrtInput(input, weights, chain)

        val convolutionsPerRow = convolutionsPerRow(input.numberColumns(), filterWidth)

        val inputGradient = collectGradients(input.numberRows(), input.numberColumns(), expandedInputGradient, convolutionsPerRow)

        val parameterGradients = differentiateProjectionWrtParameters(input, weights, optimizeWeights, bias, optimizeBias, chain)

        return BackwardResult(inputGradient, parameterGradients)

    }

    fun collectGradients(inputRows : Int, inputColumns : Int, expandedInputGradient: RealMatrix, convolutionsPerRow: Int): RealMatrix {

        val inputGradient = createRealMatrix(inputRows, inputColumns)

        for (indexConvolution in 0..expandedInputGradient.numberColumns() - 1) {

            for (indexConvolutionEntry in 0..expandedInputGradient.numberRows() - 1) {

                val inputRow = expandedRowToOriginalRow(indexConvolution, indexConvolutionEntry, convolutionsPerRow, filterWidth)
                val inputColumn = expandedColumnToOriginalColumn(indexConvolution, indexConvolutionEntry, convolutionsPerRow, filterWidth)

                val derivative = expandedInputGradient.get(indexConvolutionEntry, indexConvolution)

                inputGradient.add(inputRow, inputColumn, derivative)

            }

        }

        return inputGradient
    }

    override fun optimize(gradients: Array<RealMatrix?>) {

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

fun expandMatrixForConvolution(input: RealMatrix, filterWidth : Int, filterHeight: Int): RealMatrix {

    val convolutionsPerRow = input.numberColumns() - filterWidth + 1
    val convolutionsPerColumn = input.numberRows() - filterHeight + 1

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

fun convolutionsPerRow(numberColumns : Int, filterWidth: Int) =

    numberColumns - filterWidth + 1

fun convolutionsPerColumn(numberRows : Int, filterHeight: Int) =

    numberRows - filterHeight + 1

fun expandedRowToOriginalRow(indexConvolution : Int, indexConvolutionEntry : Int, perRow : Int, filterWidth : Int) =

    indexConvolution / perRow + indexConvolutionEntry / filterWidth

fun expandedColumnToOriginalColumn(indexConvolution: Int, indexConvolutionEntry: Int, perRow: Int, filterWidth: Int) =

    indexConvolution % perRow + indexConvolutionEntry % filterWidth
package shape.komputation.layers.feedforward.recurrent

import shape.komputation.functions.backwardProjectionWrtInput
import shape.komputation.functions.backwardProjectionWrtWeights
import shape.komputation.functions.project
import shape.komputation.layers.ContinuationLayer
import shape.komputation.layers.concatenateNames
import shape.komputation.layers.feedforward.createIdentityLayer
import shape.komputation.matrix.DoubleMatrix
import shape.komputation.optimization.*

class StepProjection(
    name : String? = null,
    private val weights : DoubleArray,
    private val numberWeightRows: Int,
    private val numberWeightColumns: Int,
    private val seriesAccumulator : DenseAccumulator? = null) : ContinuationLayer(name) {

    private var inputEntries = DoubleArray(0)
    private var numberInputRows = -1
    private var numberInputColumns = -1

    private val numberWeightEntries = numberWeightRows * numberWeightColumns

    override fun forward(input: DoubleMatrix) : DoubleMatrix {

        this.inputEntries = input.entries
        this.numberInputRows = input.numberRows
        this.numberInputColumns = input.numberColumns

        val projected = project(this.inputEntries, this.numberInputRows, this.numberInputColumns, this.weights, this.numberWeightRows, this.numberWeightColumns)

        return DoubleMatrix(this.numberWeightRows, this.numberInputColumns, projected)

    }

    override fun backward(chain : DoubleMatrix) : DoubleMatrix {

        val chainEntries = chain.entries
        val numberChainRows = chain.numberRows

        val gradient = backwardProjectionWrtInput(
            this.numberInputRows,
            this.numberInputColumns,
            this.numberInputRows * this.numberInputColumns,
            this.weights,
            this.numberWeightRows,
            chainEntries,
            numberChainRows)

        if (seriesAccumulator != null) {

            val stepDifferentiation = backwardProjectionWrtWeights(
                this.numberWeightEntries,
                this.numberWeightRows,
                this.numberWeightColumns,
                this.inputEntries,
                this.numberInputRows,
                chainEntries,
                numberChainRows,
                chain.numberColumns)

            seriesAccumulator.accumulate(stepDifferentiation)

        }

        return DoubleMatrix(this.numberInputRows, this.numberInputColumns, gradient)

    }

}

fun createStepProjection(
    name : String?,
    weights: DoubleArray,
    numberWeightRows: Int,
    numberWeightColumns: Int,
    seriesAccumulator: DenseAccumulator? = null) =

    StepProjection(name, weights, numberWeightRows, numberWeightColumns, seriesAccumulator)

fun createStepProjections(
    name : String?,
    numberSteps : Int,
    useIdentityAtFirstStep : Boolean,
    weights: DoubleArray,
    inputDimension: Int,
    outputDimension: Int,
    seriesAccumulator: DenseAccumulator) =

    Array(numberSteps) { indexStep ->

        val stateProjectionLayerName = concatenateNames(name, "step-$indexStep")

        if (useIdentityAtFirstStep && indexStep == 0) {

            createIdentityLayer(stateProjectionLayerName)

        }
        else {

            createStepProjection(stateProjectionLayerName, weights, outputDimension, inputDimension, seriesAccumulator)
        }

    }
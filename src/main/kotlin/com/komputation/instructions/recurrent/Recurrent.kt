package com.komputation.instructions.recurrent

import com.komputation.cpu.instructions.CpuContinuationInstruction
import com.komputation.cpu.layers.recurrent.CpuRecurrent
import com.komputation.cpu.layers.recurrent.Direction
import com.komputation.cpu.layers.recurrent.extraction.AllSteps
import com.komputation.cpu.layers.recurrent.extraction.LastStep
import com.komputation.initialization.InitializationStrategy
import com.komputation.initialization.initializeWeights
import com.komputation.instructions.combination.addition
import com.komputation.instructions.concatenateNames
import com.komputation.instructions.continuation.activation.Activation
import com.komputation.instructions.continuation.activation.activation
import com.komputation.instructions.continuation.projection.SharedWeighting
import com.komputation.instructions.continuation.projection.projection
import com.komputation.optimization.OptimizationInstruction

enum class ResultExtraction {
    AllSteps,
    LastStep
}

class Recurrent internal constructor(
    private val name: String?,
    private val hiddenDimension: Int,
    private val activation: Activation,
    private val direction: Direction,
    private val resultExtraction: ResultExtraction,
    private val inputWeightingInitialization: InitializationStrategy,
    private val previousStateWeightingInitialization: InitializationStrategy,
    private val biasInitialization: InitializationStrategy?,
    private val optimization: OptimizationInstruction? = null,
    private val optimizeInitialState: Boolean = false) : CpuContinuationInstruction /*, CudaContinuationInstruction */ {

    private var minimumNumberInputColumns = -1
    private var maximumNumberInputColumns = -1

    override fun setInputDimensionsFromPreviousInstruction(numberInputRows: Int, minimumNumberInputColumns: Int, maximumNumberInputColumns: Int) {
        this.minimumNumberInputColumns = minimumNumberInputColumns
        this.maximumNumberInputColumns = maximumNumberInputColumns

        this.inputProjection.setInputDimensionsFromPreviousInstruction(numberInputRows, minimumNumberInputColumns, maximumNumberInputColumns)

        this.previousStateWeighting = createPreviousStateWeighting(this.maximumNumberInputColumns)
        this.previousStateWeighting!!.setInputDimensionsFromPreviousInstruction(numberInputRows, 1, 1)

        this.additions = createAdditions(this.maximumNumberInputColumns)
        this.additions!!.setInputDimensionsFromPreviousInstruction(numberInputRows, 1, 1)

        this.activations = createActivations(this.maximumNumberInputColumns)
        this.activations!!.setInputDimensionsFromPreviousInstruction(numberInputRows, 1, 1)
    }

    override val minimumNumberOutputColumns
        get() = when (this.resultExtraction) {
            ResultExtraction.LastStep -> 1
            ResultExtraction.AllSteps -> this.minimumNumberInputColumns
        }
    override val maximumNumberOutputColumns
        get() = when (this.resultExtraction) {
            ResultExtraction.LastStep -> 1
            ResultExtraction.AllSteps -> this.maximumNumberInputColumns
        }

    override val numberOutputRows = this.hiddenDimension

    private val inputProjection = projection(
        concatenateNames(this.name, "input-weighting"),
        this.hiddenDimension,
        this.inputWeightingInitialization,
        this.biasInitialization,
        this.optimization)

    private var previousStateWeighting : ParameterizedSeries? = null
    private fun createPreviousStateWeighting(steps : Int) = parameterizedSeries(
        concatenateNames(this.name, "previous-hidden-state-weighting"),
        { initializeWeights(this.previousStateWeightingInitialization, this.hiddenDimension, this.hiddenDimension, this.hiddenDimension) },
        this.hiddenDimension,
        this.hiddenDimension,
        Array(steps-1) { index ->
            SharedWeighting(
                concatenateNames(this.name, "previous-hidden-state-weighting-$index"),
                this.hiddenDimension,
                this.optimization
            )
        },
        this.optimization)

    private var additions : CombinationSeries? = null
    private fun createAdditions(steps : Int) = combinationSeries(
        concatenateNames(this.name, "addition"),
        Array(steps-1) { index ->
            addition(concatenateNames(this.name, "addition-$index"))
        }
    )

    private var activations : Series? = null
    private fun createActivations(steps : Int) = series(
        concatenateNames(this.name, "activation"),
        Array(steps) { index ->
            activation(concatenateNames(this.name, "activation-$index"), this.activation)
        }
    )

    override fun buildForCpu() =
        CpuRecurrent(
            this.name,
            this.minimumNumberInputColumns,
            this.maximumNumberInputColumns,
            this.hiddenDimension,
            this.inputProjection.buildForCpu(),
            this.previousStateWeighting!!.buildForCpu(),
            this.additions!!.buildForCpu(),
            this.activations!!.buildForCpu(),
            this.direction,
            when(this.resultExtraction) {
                ResultExtraction.AllSteps -> AllSteps(this.hiddenDimension, this.minimumNumberInputColumns, this.maximumNumberInputColumns)
                ResultExtraction.LastStep -> LastStep(this.hiddenDimension, this.direction == Direction.RightToLeft)
            })

    /* override fun buildForCuda(context: CudaContext, cublasHandle: cublasHandle) =
        CudaRecurrent(
            this.name,
            this.maximumNumberInputColumns,
            this.hiddenDimension,
            this.inputProjection.buildForCuda(context, cublasHandle),
            this.activation) */

}

fun recurrent(
    hiddenDimension: Int,
    activation: Activation,
    resultExtraction: ResultExtraction,
    direction: Direction,
    initialization: InitializationStrategy,
    optimization: OptimizationInstruction? = null) =
    recurrent(null, hiddenDimension, activation, resultExtraction, direction, initialization, initialization, initialization, optimization)

fun recurrent(
    hiddenDimension: Int,
    activation: Activation,
    resultExtraction: ResultExtraction,
    direction: Direction,
    inputWeightingInitialization: InitializationStrategy,
    previousStateWeightingInitialization: InitializationStrategy = inputWeightingInitialization,
    biasInitialization: InitializationStrategy? = inputWeightingInitialization,
    optimization: OptimizationInstruction? = null) =
    recurrent(null, hiddenDimension, activation, resultExtraction, direction, inputWeightingInitialization, previousStateWeightingInitialization, biasInitialization, optimization)

fun recurrent(
    name : String,
    hiddenDimension: Int,
    activation: Activation,
    resultExtraction: ResultExtraction,
    direction: Direction,
    initialization: InitializationStrategy,
    optimization: OptimizationInstruction? = null) =
    recurrent(name, hiddenDimension, activation, resultExtraction, direction, initialization, initialization, initialization, optimization)

fun recurrent(
    name: String? = null,
    hiddenDimension: Int,
    activation: Activation,
    resultExtraction: ResultExtraction,
    direction: Direction,
    inputWeightingInitialization: InitializationStrategy,
    previousStateWeightingInitialization: InitializationStrategy = inputWeightingInitialization,
    biasInitialization: InitializationStrategy? = inputWeightingInitialization,
    optimization: OptimizationInstruction? = null) =
    Recurrent(name, hiddenDimension, activation, direction, resultExtraction, inputWeightingInitialization, previousStateWeightingInitialization, biasInitialization, optimization)
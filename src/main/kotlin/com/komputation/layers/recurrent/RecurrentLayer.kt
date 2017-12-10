package com.komputation.layers.recurrent

import com.komputation.cpu.layers.combination.CpuAdditionCombination
import com.komputation.cpu.layers.forward.projection.CpuBiasLayer
import com.komputation.cpu.layers.forward.projection.CpuWeightingLayer
import com.komputation.cpu.layers.recurrent.CpuRecurrentLayer
import com.komputation.cpu.layers.recurrent.Direction
import com.komputation.cpu.layers.recurrent.extraction.AllSteps
import com.komputation.cpu.layers.recurrent.extraction.LastStep
import com.komputation.cpu.layers.recurrent.series.ParameterizedSeries
import com.komputation.cpu.layers.recurrent.series.Series
import com.komputation.cpu.optimization.DenseAccumulator
import com.komputation.initialization.InitializationStrategy
import com.komputation.initialization.initializeColumnVector
import com.komputation.initialization.initializeWeights
import com.komputation.layers.CpuForwardLayerInstruction
import com.komputation.layers.concatenateNames
import com.komputation.layers.forward.activation.ActivationFunction
import com.komputation.layers.forward.activation.activationLayer
import com.komputation.layers.forward.projection.weightingLayer
import com.komputation.optimization.OptimizationInstruction

enum class ResultExtraction {
    AllSteps,
    LastStep
}

class RecurrentLayer internal constructor(
    private val name: String?,
    private val maximumSteps: Int,
    private val hasFixedLength: Boolean,
    private val inputDimension: Int,
    private val hiddenDimension: Int,
    private val activation: ActivationFunction,
    private val direction: Direction,
    private val resultExtraction: ResultExtraction,
    private val inputWeightingInitialization: InitializationStrategy,
    private val previousStateWeightingInitialization: InitializationStrategy,
    private val biasInitialization: InitializationStrategy?,
    private val optimization: OptimizationInstruction? = null) : CpuForwardLayerInstruction {

    override fun buildForCpu(): CpuRecurrentLayer {
        val minimumSteps = if (this.hasFixedLength) this.maximumSteps else 1

        val inputWeightingLayerName = concatenateNames(this.name, "input-weighting")
        val inputWeightingLayer = weightingLayer(inputWeightingLayerName, this.inputDimension, this.maximumSteps, this.hasFixedLength, this.hiddenDimension, this.inputWeightingInitialization, this.optimization).buildForCpu()

        val initialState = FloatArray(this.hiddenDimension)

        val previousStateWeights = initializeWeights(this.previousStateWeightingInitialization, this.hiddenDimension, this.hiddenDimension, this.hiddenDimension)
        val previousStateWeightingName= concatenateNames(this.name, "previous-hidden-state-weighting")

        val weightSeriesAccumulator = DenseAccumulator(previousStateWeights.size)
        val previousStateWeighting = ParameterizedSeries(
            previousStateWeightingName,
            Array(this.maximumSteps) { index ->
                CpuWeightingLayer(
                    concatenateNames(previousStateWeightingName, index.toString()),
                    this.hiddenDimension,
                    1,
                    1,
                    this.hiddenDimension,
                    previousStateWeights,
                    weightSeriesAccumulator)
            },
            previousStateWeights,
            weightSeriesAccumulator,
            DenseAccumulator(previousStateWeights.size),
            this.optimization?.buildForCpu()?.invoke(this.hiddenDimension, this.hiddenDimension))

        val bias =
            if (this.biasInitialization != null) {
                val biasName = concatenateNames(this.name, "bias")
                val bias = initializeColumnVector(this.biasInitialization, this.hiddenDimension)

                val biasAccumulator = DenseAccumulator(bias.size)
                ParameterizedSeries(
                    biasName,
                    Array(this.maximumSteps) { index ->
                        CpuBiasLayer(
                            concatenateNames(biasName, index.toString()),
                            this.hiddenDimension,
                            1,
                            1,
                            bias,
                            biasAccumulator)
                    },
                    bias,
                    biasAccumulator,
                    DenseAccumulator(bias.size),
                    this.optimization?.buildForCpu()?.invoke(this.hiddenDimension, 1))
            }
            else {
                null
            }

        val additions = Array(this.maximumSteps) { index ->
            val additionName = concatenateNames(this.name, "addition-$index")
            CpuAdditionCombination(additionName, this.hiddenDimension, 1)
        }

        val activation = Series(
            concatenateNames(this.name, "activation"),
            Array(this.maximumSteps) { index ->
                val activationName = concatenateNames(this.name, "activation-$index")
                activationLayer(activationName, this.activation, this.hiddenDimension, 1, this.hasFixedLength).buildForCpu()
            }
        )

        val resultExtraction = when(this.resultExtraction) {
            ResultExtraction.AllSteps -> AllSteps(activation, this.hiddenDimension, minimumSteps, this.maximumSteps)
            ResultExtraction.LastStep -> LastStep(activation, this.hiddenDimension, this.direction == Direction.Backward)
        }

        val recurrentLayer = CpuRecurrentLayer(
            this.name,
            minimumSteps,
            this.maximumSteps,
            this.hiddenDimension,
            inputWeightingLayer,
            direction,
            resultExtraction,
            initialState,
            previousStateWeighting,
            additions,
            bias,
            activation)

        return recurrentLayer
    }
}

fun recurrentLayer(
    maximumSteps: Int,
    hasFixedLength: Boolean,
    inputDimension: Int,
    hiddenDimension: Int,
    direction: Direction,
    resultExtraction: ResultExtraction,
    activation: ActivationFunction,
    initialization: InitializationStrategy,
    optimization: OptimizationInstruction? = null) =
    recurrentLayer(null, maximumSteps, hasFixedLength, inputDimension, hiddenDimension, activation, direction, resultExtraction, initialization, optimization)

fun recurrentLayer(
    name: String? = null,
    maximumSteps: Int,
    hasFixedLength: Boolean,
    inputDimension: Int,
    hiddenDimension: Int,
    activation: ActivationFunction,
    direction: Direction,
    resultExtraction: ResultExtraction,
    initialization: InitializationStrategy,
    optimization: OptimizationInstruction? = null) =
    RecurrentLayer(name, maximumSteps, hasFixedLength, inputDimension, hiddenDimension, activation, direction, resultExtraction, initialization, initialization, initialization, optimization)

fun recurrentLayer(
    maximumSteps: Int,
    hasFixedLength: Boolean,
    inputDimension: Int,
    hiddenDimension: Int,
    direction: Direction,
    resultExtraction: ResultExtraction,
    activation: ActivationFunction,
    inputWeightingInitialization: InitializationStrategy,
    previousStateWeightingInitialization: InitializationStrategy = inputWeightingInitialization,
    biasInitialization: InitializationStrategy? = inputWeightingInitialization,
    optimization: OptimizationInstruction? = null) =
    recurrentLayer(null, maximumSteps, hasFixedLength, inputDimension, hiddenDimension, activation, direction, resultExtraction, inputWeightingInitialization, previousStateWeightingInitialization, biasInitialization, optimization)

fun recurrentLayer(
    name: String? = null,
    maximumSteps: Int,
    hasFixedLength: Boolean,
    inputDimension: Int,
    hiddenDimension: Int,
    activation: ActivationFunction,
    direction: Direction,
    resultExtraction: ResultExtraction,
    inputWeightingInitialization: InitializationStrategy,
    previousStateWeightingInitialization: InitializationStrategy = inputWeightingInitialization,
    biasInitialization: InitializationStrategy? = inputWeightingInitialization,
    optimization: OptimizationInstruction? = null) =
    RecurrentLayer(name, maximumSteps, hasFixedLength, inputDimension, hiddenDimension, activation, direction, resultExtraction, inputWeightingInitialization, previousStateWeightingInitialization, biasInitialization, optimization)
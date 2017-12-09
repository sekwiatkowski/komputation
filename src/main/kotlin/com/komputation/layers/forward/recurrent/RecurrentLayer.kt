package com.komputation.layers.forward.recurrent

import com.komputation.cpu.layers.combination.CpuAdditionCombination
import com.komputation.cpu.layers.forward.projection.CpuBiasLayer
import com.komputation.cpu.layers.forward.projection.CpuWeightingLayer
import com.komputation.cpu.layers.recurrent.CpuRecurrentLayer
import com.komputation.cpu.layers.recurrent.ParameterizedSeries
import com.komputation.cpu.layers.recurrent.Series
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

class RecurrentLayer internal constructor(
    private val name : String?,
    private val maximumSteps : Int,
    private val hasFixedLength : Boolean,
    private val inputDimension : Int,
    private val hiddenDimension : Int,
    private val weightInitialization: InitializationStrategy,
    private val biasInitialization: InitializationStrategy?,
    private val activation : ActivationFunction,
    private val optimization: OptimizationInstruction? = null) : CpuForwardLayerInstruction {

    override fun buildForCpu(): CpuRecurrentLayer {
        val inputWeightingLayerName = concatenateNames(this.name, "input-weighting")
        val inputWeightingLayer = weightingLayer(inputWeightingLayerName, this.inputDimension, this.maximumSteps, this.hasFixedLength, this.hiddenDimension, this.weightInitialization, this.optimization).buildForCpu()

        val initialState = FloatArray(this.hiddenDimension)

        val weights = initializeWeights(this.weightInitialization, this.hiddenDimension, this.hiddenDimension, this.hiddenDimension)
        val previousHiddenStateWeightingName= concatenateNames(this.name, "previous-hidden-state-weighting")

        val weightSeriesAccumulator = DenseAccumulator(weights.size)
        val previousHiddenStateWeighting = ParameterizedSeries(
            previousHiddenStateWeightingName,
            Array(this.maximumSteps) { index ->
                CpuWeightingLayer(
                    concatenateNames(previousHiddenStateWeightingName, index.toString()),
                    this.hiddenDimension,
                    1,
                    1,
                    this.hiddenDimension,
                    weights,
                    weightSeriesAccumulator)
            },
            weights,
            weightSeriesAccumulator,
            DenseAccumulator(weights.size),
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

        val recurrentLayer = CpuRecurrentLayer(
            this.name,
            if (this.hasFixedLength) this.maximumSteps else 1,
            this.maximumSteps,
            this.hiddenDimension,
            inputWeightingLayer,
            initialState,
            previousHiddenStateWeighting,
            additions,
            bias,
            activation)

        return recurrentLayer
    }
}

fun recurrentLayer(
    maximumSteps: Int,
    hasFixedLength: Boolean,
    inputDimension : Int,
    hiddenDimension: Int,
    weightInitialization: InitializationStrategy,
    biasInitialization: InitializationStrategy?,
    activation : ActivationFunction,
    optimization: OptimizationInstruction? = null) =
    recurrentLayer(null, maximumSteps, hasFixedLength, inputDimension, hiddenDimension, weightInitialization, biasInitialization, activation, optimization)

fun recurrentLayer(
    name : String? = null,
    maximumSteps : Int,
    hasFixedLength: Boolean,
    inputDimension: Int,
    hiddenDimension: Int,
    weightInitialization: InitializationStrategy,
    biasInitialization: InitializationStrategy?,
    activation : ActivationFunction,
    optimization: OptimizationInstruction? = null) =
    RecurrentLayer(name, maximumSteps, hasFixedLength, inputDimension, hiddenDimension, weightInitialization, biasInitialization, activation, optimization)
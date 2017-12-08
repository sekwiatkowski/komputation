package com.komputation.layers.forward.recurrent

import com.komputation.cpu.layers.combination.CpuAdditionCombination
import com.komputation.cpu.layers.forward.projection.CpuWeightingLayer
import com.komputation.cpu.layers.recurrent.CpuRecurrentLayer
import com.komputation.cpu.layers.recurrent.Series
import com.komputation.cpu.optimization.DenseAccumulator
import com.komputation.initialization.InitializationStrategy
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
    private val initialization: InitializationStrategy,
    private val activation : ActivationFunction,
    private val optimization: OptimizationInstruction? = null) : CpuForwardLayerInstruction {

    override fun buildForCpu(): CpuRecurrentLayer {
        val inputWeightingLayerName = concatenateNames(this.name, "input-weighting")
        val inputWeightingLayer = weightingLayer(inputWeightingLayerName, this.inputDimension, this.maximumSteps, this.hasFixedLength, this.hiddenDimension, this.initialization, this.optimization).buildForCpu()

        val initialState = FloatArray(this.hiddenDimension)

        val weights = initializeWeights(this.initialization, this.hiddenDimension, this.hiddenDimension, this.hiddenDimension)
        val numberWeightEntries = weights.size
        val updateRule = this.optimization?.buildForCpu()?.invoke(this.hiddenDimension, this.hiddenDimension)

        val previousHiddenStateWeightingLayerName= concatenateNames(this.name, "previous-hidden-state-weighting")
        val seriesAccumulator = DenseAccumulator(numberWeightEntries)
        val previousHiddenStateWeighting = Series(
            previousHiddenStateWeightingLayerName,
            weights,
            Array(this.maximumSteps) { index ->
                CpuWeightingLayer(
                    concatenateNames(previousHiddenStateWeightingLayerName, index.toString()),
                    weights,
                    this.hiddenDimension,
                    1,
                    1,
                    this.hiddenDimension,
                    seriesAccumulator)
            },
            seriesAccumulator,
            DenseAccumulator(numberWeightEntries),
            updateRule)

        val additions = Array(this.maximumSteps) { index ->
            val additionName = concatenateNames(this.name, "addition-$index")
            CpuAdditionCombination(additionName, this.hiddenDimension, 1)
        }

        val activations = Array(this.maximumSteps) { index ->
            val activationName = concatenateNames(this.name, "activation-$index")
            activationLayer(activationName, this.activation, this.hiddenDimension, 1, this.hasFixedLength).buildForCpu()
        }

        return CpuRecurrentLayer(this.name, if(this.hasFixedLength) this.maximumSteps else 1, this.maximumSteps, this.hiddenDimension, inputWeightingLayer, initialState, previousHiddenStateWeighting, additions, activations)
    }

}

fun recurrentLayer(maximumSteps: Int, hasFixedLength: Boolean, inputDimension : Int, hiddenDimension: Int, initialization: InitializationStrategy, activation : ActivationFunction, optimization: OptimizationInstruction? = null) =
    recurrentLayer(null, maximumSteps, hasFixedLength, inputDimension, hiddenDimension, initialization, activation, optimization)

fun recurrentLayer(name : String? = null, maximumSteps : Int, hasFixedLength: Boolean, inputDimension: Int, hiddenDimension: Int, initialization: InitializationStrategy, activation : ActivationFunction, optimization: OptimizationInstruction? = null) =
    RecurrentLayer(name, maximumSteps, hasFixedLength, inputDimension, hiddenDimension, initialization, activation, optimization)
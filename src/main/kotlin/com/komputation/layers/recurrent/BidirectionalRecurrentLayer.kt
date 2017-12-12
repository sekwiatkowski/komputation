package com.komputation.layers.recurrent

import com.komputation.cpu.layers.recurrent.Direction
import com.komputation.initialization.InitializationStrategy
import com.komputation.layers.forward.activation.ActivationFunction
import com.komputation.layers.forward.concatenation
import com.komputation.optimization.OptimizationInstruction

fun bidirectionalRecurrentLayer(
    maximumSteps : Int,
    hasFixedLength : Boolean,
    inputDimension : Int,
    hiddenDimension : Int,
    activation: ActivationFunction,
    resultExtraction: ResultExtraction,
    initialization: InitializationStrategy,
    optimization : OptimizationInstruction? = null) =

    bidirectionalRecurrentLayer(
        null,
        maximumSteps,
        hasFixedLength,
        inputDimension,
        hiddenDimension,
        activation,
        resultExtraction,
        initialization,
        optimization)

fun bidirectionalRecurrentLayer(
    name : String?,
    maximumSteps : Int,
    hasFixedLength : Boolean,
    inputDimension : Int,
    hiddenDimension : Int,
    activation: ActivationFunction,
    resultExtraction: ResultExtraction,
    initialization: InitializationStrategy,
    optimization : OptimizationInstruction? = null) =

    bidirectionalRecurrentLayer(
        name,
        maximumSteps,
        hasFixedLength,
        inputDimension,
        hiddenDimension,
        activation,
        resultExtraction,
        initialization,
        initialization,
        initialization,
        optimization)

fun bidirectionalRecurrentLayer(
    maximumSteps : Int,
    hasFixedLength : Boolean,
    inputDimension : Int,
    hiddenDimension : Int,
    activation: ActivationFunction,
    resultExtraction: ResultExtraction,
    inputWeightingInitialization: InitializationStrategy,
    previousStateInitialization: InitializationStrategy = inputWeightingInitialization,
    biasInitialization: InitializationStrategy = inputWeightingInitialization,
    optimization : OptimizationInstruction? = null) =

    bidirectionalRecurrentLayer(
        null,
        maximumSteps,
        hasFixedLength,
        inputDimension,
        hiddenDimension,
        activation,
        resultExtraction,
        inputWeightingInitialization,
        previousStateInitialization,
        biasInitialization,
        optimization)

fun bidirectionalRecurrentLayer(
    name : String?,
    maximumSteps : Int,
    hasFixedLength : Boolean,
    inputDimension : Int,
    hiddenDimension : Int,
    activation: ActivationFunction,
    resultExtraction: ResultExtraction,
    inputWeightingInitialization: InitializationStrategy,
    previousStateInitialization: InitializationStrategy = inputWeightingInitialization,
    biasInitialization: InitializationStrategy = inputWeightingInitialization,
    optimization : OptimizationInstruction? = null) =

    concatenation(
        name,
        recurrentLayer(name, maximumSteps, hasFixedLength, inputDimension, hiddenDimension, activation, resultExtraction, Direction.Forward, inputWeightingInitialization, previousStateInitialization, biasInitialization, optimization),
        recurrentLayer(name, maximumSteps, hasFixedLength, inputDimension, hiddenDimension, activation, resultExtraction, Direction.Backward, inputWeightingInitialization, previousStateInitialization, biasInitialization, optimization)
    )
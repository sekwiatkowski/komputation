package com.komputation.instructions.recurrent

import com.komputation.cpu.layers.recurrent.Direction
import com.komputation.initialization.InitializationStrategy
import com.komputation.instructions.continuation.activation.Activation
import com.komputation.instructions.continuation.concatenation.concatenation
import com.komputation.optimization.OptimizationInstruction

fun bidirectionalRecurrentLayer(
    hiddenDimension : Int,
    activation: Activation,
    resultExtraction: ResultExtraction,
    initialization: InitializationStrategy,
    optimization : OptimizationInstruction? = null) =

    bidirectionalRecurrentLayer(
        hiddenDimension,
        activation,
        resultExtraction,
        initialization,
        initialization,
        initialization,
        optimization)

fun bidirectionalRecurrentLayer(
    name : String,
    hiddenDimension : Int,
    activation: Activation,
    resultExtraction: ResultExtraction,
    initialization: InitializationStrategy,
    optimization : OptimizationInstruction? = null) =

    bidirectionalRecurrentLayer(
        name,
        hiddenDimension,
        activation,
        resultExtraction,
        initialization,
        initialization,
        initialization,
        optimization)

fun bidirectionalRecurrentLayer(
    hiddenDimension : Int,
    activation: Activation,
    resultExtraction: ResultExtraction,
    inputWeightingInitialization: InitializationStrategy,
    previousStateInitialization: InitializationStrategy,
    biasInitialization: InitializationStrategy,
    optimization : OptimizationInstruction? = null) =

    bidirectionalRecurrentLayer(
        null,
        hiddenDimension,
        activation,
        resultExtraction,
        inputWeightingInitialization,
        previousStateInitialization,
        biasInitialization,
        optimization)

fun bidirectionalRecurrentLayer(
    name : String?,
    hiddenDimension : Int,
    activation: Activation,
    resultExtraction: ResultExtraction,
    inputWeightingInitialization: InitializationStrategy,
    previousStateInitialization: InitializationStrategy,
    biasInitialization: InitializationStrategy,
    optimization : OptimizationInstruction? = null) =

    concatenation(
        name,
        recurrent(name, hiddenDimension, activation, resultExtraction, Direction.LeftToRight, inputWeightingInitialization, previousStateInitialization, biasInitialization, optimization),
        recurrent(name, hiddenDimension, activation, resultExtraction, Direction.RightToLeft, inputWeightingInitialization, previousStateInitialization, biasInitialization, optimization)
    )
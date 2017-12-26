package com.komputation.instructions.recurrent

import com.komputation.cpu.layers.recurrent.Direction
import com.komputation.initialization.InitializationStrategy
import com.komputation.instructions.continuation.activation.Activation
import com.komputation.instructions.continuation.stack.stack
import com.komputation.optimization.OptimizationInstruction

fun bidirectionalRecurrent(
    hiddenDimension : Int,
    activation: Activation,
    resultExtraction: ResultExtraction,
    initialization: InitializationStrategy,
    optimization : OptimizationInstruction? = null) =

    bidirectionalRecurrent(
        hiddenDimension,
        activation,
        resultExtraction,
        initialization,
        initialization,
        initialization,
        optimization)

fun bidirectionalRecurrent(
    name : String,
    hiddenDimension : Int,
    activation: Activation,
    resultExtraction: ResultExtraction,
    initialization: InitializationStrategy,
    optimization : OptimizationInstruction? = null) =

    bidirectionalRecurrent(
        name,
        hiddenDimension,
        activation,
        resultExtraction,
        initialization,
        initialization,
        initialization,
        optimization)

fun bidirectionalRecurrent(
    hiddenDimension : Int,
    activation: Activation,
    resultExtraction: ResultExtraction,
    inputWeightingInitialization: InitializationStrategy,
    previousStateInitialization: InitializationStrategy,
    biasInitialization: InitializationStrategy,
    optimization : OptimizationInstruction? = null) =

    bidirectionalRecurrent(
        null,
        hiddenDimension,
        activation,
        resultExtraction,
        inputWeightingInitialization,
        previousStateInitialization,
        biasInitialization,
        optimization)

fun bidirectionalRecurrent(
    name : String?,
    hiddenDimension : Int,
    activation: Activation,
    resultExtraction: ResultExtraction,
    inputWeightingInitialization: InitializationStrategy,
    previousStateInitialization: InitializationStrategy,
    biasInitialization: InitializationStrategy,
    optimization : OptimizationInstruction? = null) =

    stack(
        name,
        recurrent(name, hiddenDimension, activation, resultExtraction, Direction.LeftToRight, inputWeightingInitialization, previousStateInitialization, biasInitialization, optimization),
        recurrent(name, hiddenDimension, activation, resultExtraction, Direction.RightToLeft, inputWeightingInitialization, previousStateInitialization, biasInitialization, optimization)
    )
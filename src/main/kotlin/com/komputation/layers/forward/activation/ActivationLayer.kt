package com.komputation.layers.forward.activation

import com.komputation.layers.CpuActivationLayerInstruction
import com.komputation.layers.CudaActivationLayerInstruction

fun activationLayer(name: String?, function: ActivationFunction, numberRows : Int, maximumColumns : Int, hasFixedLength : Boolean) : ActivationLayerInstruction =
    when (function) {
        ActivationFunction.Identity ->
            identityLayer(name, numberRows, maximumColumns)
        ActivationFunction.ReLU ->
            reluLayer(name, numberRows, maximumColumns, hasFixedLength)
        ActivationFunction.Sigmoid ->
            sigmoidLayer(name, numberRows, maximumColumns, hasFixedLength)
        ActivationFunction.Softmax ->
            softmaxLayer(name, numberRows, maximumColumns, hasFixedLength)
        ActivationFunction.Tanh ->
            tanhLayer(name, numberRows, maximumColumns, hasFixedLength)
    }

interface ActivationLayerInstruction : CpuActivationLayerInstruction, CudaActivationLayerInstruction
package com.komputation.layers.forward.activation

import com.komputation.layers.CpuActivationLayerInstruction
import com.komputation.layers.CudaActivationLayerInstruction

fun activationLayer(name: String?, function: ActivationFunction, numberRows : Int, numberColumns : Int) : ActivationLayerInstruction =

    when (function) {

        ActivationFunction.Identity ->
            identityLayer(name, numberRows, numberColumns)
        ActivationFunction.ReLU ->
            reluLayer(name, numberRows, numberColumns)
        ActivationFunction.Sigmoid ->
            sigmoidLayer(name, numberRows, numberColumns)
        ActivationFunction.Softmax ->
            softmaxLayer(name, numberRows, numberColumns)
        ActivationFunction.Tanh ->
            tanhLayer(name, numberRows, numberColumns)

    }

interface ActivationLayerInstruction : CpuActivationLayerInstruction, CudaActivationLayerInstruction
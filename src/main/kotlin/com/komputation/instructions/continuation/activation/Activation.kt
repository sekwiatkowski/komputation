package com.komputation.instructions.continuation.activation

import com.komputation.instructions.continuation.ActivationInstruction

fun activation(name: String?, function: Activation) : ActivationInstruction =
    when (function) {
        Activation.Identity ->
            identityLayer(name)
        Activation.ReLU ->
            relu(name)
        Activation.Sigmoid ->
            sigmoid(name)
        Activation.Softmax ->
            softmax(name)
        Activation.Tanh ->
            tanh(name)
    }

enum class Activation {
    Identity,
    ReLU,
    Sigmoid,
    Softmax,
    Tanh
}
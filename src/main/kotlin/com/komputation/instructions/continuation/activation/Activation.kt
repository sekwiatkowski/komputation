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
fun recurrentActivation(name: String?, function: RecurrentActivation) : ActivationInstruction =
    when (function) {
        RecurrentActivation.Identity ->
            identityLayer(name)
        RecurrentActivation.ReLU ->
            relu(name)
        RecurrentActivation.Sigmoid ->
            sigmoid(name)
        RecurrentActivation.Tanh ->
            tanh(name)
    }

enum class RecurrentActivation(val id : Int) {
    Identity(0),
    ReLU(1),
    Sigmoid(2),
    Tanh(3)
}
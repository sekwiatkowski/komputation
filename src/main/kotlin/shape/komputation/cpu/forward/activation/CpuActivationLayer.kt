package shape.komputation.cpu.forward.activation

import shape.komputation.cpu.CpuForwardLayer
import shape.komputation.functions.activation.ActivationFunction
import shape.komputation.layers.CpuActivationLayerInstruction
import shape.komputation.layers.forward.activation.*

fun activationLayer(name: String?, function: ActivationFunction) : CpuActivationLayerInstruction =

    when (function) {

        ActivationFunction.ReLU ->
            reluLayer(name)
        ActivationFunction.Identity ->
            identityLayer(name)
        ActivationFunction.Sigmoid ->
            sigmoidLayer(name)
        ActivationFunction.Softmax ->
            softmaxLayer(name)
        ActivationFunction.Tanh ->
            tanhLayer(name)

    }

interface CpuActivationLayer : CpuForwardLayer

package shape.komputation.cpu.layers.forward.activation

import shape.komputation.cpu.layers.CpuForwardLayer
import shape.komputation.layers.CpuActivationLayerInstruction
import shape.komputation.layers.forward.activation.*

fun cpuActivationLayer(name: String?, function: ActivationFunction, dimension : Int) : CpuActivationLayerInstruction =

    when (function) {

        ActivationFunction.ReLU ->
            reluLayer(name)
        ActivationFunction.Identity ->
            identityLayer(name)
        ActivationFunction.Sigmoid ->
            sigmoidLayer(name, dimension)
        ActivationFunction.Softmax ->
            softmaxLayer(name)
        ActivationFunction.Tanh ->
            tanhLayer(name)

    }

interface CpuActivationLayer : CpuForwardLayer

package shape.komputation.cpu.layers.forward.activation

import shape.komputation.cpu.layers.CpuForwardLayer
import shape.komputation.layers.CpuActivationLayerInstruction
import shape.komputation.layers.forward.activation.*

fun cpuActivationLayer(name: String?, function: ActivationFunction, numberRows : Int, numberColumns : Int) : CpuActivationLayerInstruction =

    when (function) {

        ActivationFunction.Identity ->
            identityLayer(name)
        ActivationFunction.ReLU ->
            reluLayer(name, numberRows * numberColumns)
        ActivationFunction.Sigmoid ->
            sigmoidLayer(name, numberRows * numberColumns)
        ActivationFunction.Softmax ->
            softmaxLayer(name, numberRows, numberColumns)
        ActivationFunction.Tanh ->
            tanhLayer(name)

    }

interface CpuActivationLayer : CpuForwardLayer

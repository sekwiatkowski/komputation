package shape.komputation.layers.forward.activation

import shape.komputation.functions.activation.ActivationFunction
import shape.komputation.layers.BaseForwardLayer
import shape.komputation.layers.concatenateNames
import shape.komputation.layers.forward.identityLayer

abstract class ActivationLayer(name : String?) : BaseForwardLayer(name)

fun activationLayers(number: Int, name: String?, function: ActivationFunction) =

    Array(number) { index ->

        activationLayer(concatenateNames(name, index.toString()), function)

    }

fun activationLayer(name: String?, function: ActivationFunction) =

    when (function) {

        ActivationFunction.ReLU ->
            reluLayer(name).buildForCpu()
        ActivationFunction.Identity ->
            identityLayer(name).buildForCpu()
        ActivationFunction.Sigmoid ->
            sigmoidLayer(name).buildForCpu()
        ActivationFunction.Softmax ->
            softmaxLayer(name).buildForCpu()
        ActivationFunction.Tanh ->
            tanhLayer(name).buildForCpu()

    }

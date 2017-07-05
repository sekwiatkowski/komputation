package shape.komputation.layers.forward.activation

import shape.komputation.functions.activation.ActivationFunction
import shape.komputation.layers.ForwardLayer
import shape.komputation.layers.concatenateNames
import shape.komputation.layers.forward.identityLayer

abstract class ActivationLayer(name : String?) : ForwardLayer(name)

fun activationLayers(number: Int, name: String?, function: ActivationFunction) =

    Array(number) { index ->

        activationLayer(concatenateNames(name, index.toString()), function)

    }

fun activationLayer(name: String?, function: ActivationFunction) =

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

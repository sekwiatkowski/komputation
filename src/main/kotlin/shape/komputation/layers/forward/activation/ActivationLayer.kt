package shape.komputation.layers.forward.activation

import shape.komputation.functions.activation.ActivationFunction
import shape.komputation.layers.ForwardLayer
import shape.komputation.layers.concatenateNames
import shape.komputation.layers.forward.IdentityLayer

abstract class ActivationLayer(name : String?) : ForwardLayer(name)

fun createActivationLayers(number: Int, name: String?, function: ActivationFunction) =

    Array(number) { index ->

        createActivationLayer(concatenateNames(name, index.toString()), function)

    }

fun createActivationLayer(name: String?, function: ActivationFunction) =

    when (function) {

        ActivationFunction.ReLU ->
            ReluLayer(name)
        ActivationFunction.Identity ->
            IdentityLayer(name)
        ActivationFunction.Sigmoid ->
            SigmoidLayer(name)
        ActivationFunction.Softmax ->
            SoftmaxLayer(name)
        ActivationFunction.Tanh ->
            TanhLayer(name)

    }

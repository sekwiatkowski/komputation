package shape.komputation.layers.feedforward.activation

import shape.komputation.functions.activation.ActivationFunction
import shape.komputation.layers.ContinuationLayer
import shape.komputation.layers.concatenateNames
import shape.komputation.layers.feedforward.IdentityLayer

abstract class ActivationLayer(name : String?) : ContinuationLayer(name)

fun createActivationLayers(number: Int, name: String?, function: ActivationFunction) =

    Array(number) { index ->

        createActivationLayer(name, function)

    }

fun createActivationLayer(name: String?, function: ActivationFunction) : ActivationLayer {

    val layerName = concatenateNames(name, function.layerName)

    return when (function) {

        ActivationFunction.ReLU ->
            ReluLayer(layerName)
        ActivationFunction.Identity ->
            IdentityLayer(layerName)
        ActivationFunction.Sigmoid ->
            SigmoidLayer(layerName)
        ActivationFunction.Softmax ->
            SoftmaxLayer(layerName)
        ActivationFunction.Tanh ->
            TanhLayer(layerName)

    }

}
package shape.komputation.layers.feedforward.activation

import shape.komputation.functions.activation.ActivationFunction
import shape.komputation.layers.FeedForwardLayer

abstract class ActivationLayer(name : String? = null) : FeedForwardLayer(name)

fun createActivationLayers(name: String?, number: Int, function: ActivationFunction) =

    Array(number) { index ->

        val activationLayerName = if (name == null) null else "$name-$index"

        when (function) {

            ActivationFunction.Sigmoid -> SigmoidLayer(activationLayerName)
            ActivationFunction.ReLU -> ReluLayer(activationLayerName)
            ActivationFunction.Softmax -> SoftmaxLayer(activationLayerName)

        }

    }
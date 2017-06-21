package shape.komputation.layers.feedforward.activation

import shape.komputation.functions.activation.ActivationFunction
import shape.komputation.layers.ContinuationLayer
import shape.komputation.layers.feedforward.IdentityLayer

interface ActivationLayer

fun createActivationLayers(name: String?, number: Int, function: ActivationFunction) =

    Array<ContinuationLayer>(number) { index ->

        val activationLayerName = if (name == null) null else "$name-$index"

        when (function) {

            ActivationFunction.Sigmoid -> SigmoidLayer(activationLayerName)
            ActivationFunction.ReLU -> ReluLayer(activationLayerName)
            ActivationFunction.Softmax -> SoftmaxLayer(activationLayerName)
            ActivationFunction.Identity -> IdentityLayer(activationLayerName)

        }

    }
package shape.komputation.layers.feedforward.activation

import shape.komputation.functions.activation.ActivationFunction
import shape.komputation.layers.ContinuationLayer
import shape.komputation.layers.feedforward.IdentityLayer

interface ActivationLayer

fun createActivationLayers(number: Int, name: String?, function: ActivationFunction) =

    Array<ContinuationLayer>(number) { index ->

        val activationLayerName = if (name == null) null else "$name-$index"

        createActivationLayer(activationLayerName, function)

    }

fun createActivationLayer(name: String?, function: ActivationFunction) : ContinuationLayer =


    when (function) {

        ActivationFunction.Sigmoid -> SigmoidLayer(name)
        ActivationFunction.ReLU -> ReluLayer(name)
        ActivationFunction.Softmax -> SoftmaxLayer(name)
        ActivationFunction.Identity -> IdentityLayer(name)

    }
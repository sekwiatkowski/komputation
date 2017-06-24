package shape.komputation.layers.feedforward.activation

import shape.komputation.functions.activation.ActivationFunction
import shape.komputation.layers.ContinuationLayer
import shape.komputation.layers.feedforward.IdentityLayer

abstract class ActivationLayer(name : String?) : ContinuationLayer(name)

fun createActivationLayers(number: Int, name: String?, function: ActivationFunction) =

    Array(number) { index ->

        val activationLayerName = if (name == null) null else "$name-$index"

        createActivationLayer(activationLayerName, function)

    }

fun createActivationLayer(name: String?, function: ActivationFunction) : ActivationLayer =


    when (function) {

        ActivationFunction.ReLU -> ReluLayer(name)
        ActivationFunction.Identity -> IdentityLayer(name)
        ActivationFunction.Sigmoid -> SigmoidLayer(name)
        ActivationFunction.Softmax -> SoftmaxLayer(name)
        ActivationFunction.Tanh -> TanhLayer(name)

    }
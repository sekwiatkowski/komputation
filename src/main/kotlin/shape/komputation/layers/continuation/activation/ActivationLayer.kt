package shape.komputation.layers.continuation.activation

import shape.komputation.layers.continuation.ContinuationLayer

abstract class ActivationLayer(name : String? = null, numberResults: Int, numberParameters: Int) : ContinuationLayer(name, numberResults, numberParameters)
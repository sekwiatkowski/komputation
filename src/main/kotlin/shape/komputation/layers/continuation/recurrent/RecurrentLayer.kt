package shape.komputation.layers.continuation.recurrent

import shape.komputation.layers.continuation.ContinuationLayer
import shape.komputation.layers.continuation.OptimizableContinuationLayer
import shape.komputation.layers.continuation.activation.ActivationLayer
import shape.komputation.matrix.RealMatrix

class RecurrentLayer(
    name : String?,
    private val statefulProjectionLayer: StatefulProjectionLayer,
    private val activationLayer: ActivationLayer) : ContinuationLayer(name), OptimizableContinuationLayer {

    // activate(stateful_projection(input))
    override fun forward(input : RealMatrix) : RealMatrix {

        val projection = this.statefulProjectionLayer.forward(input)

        val activation = this.activationLayer.forward(projection)

        return activation

    }

    override fun backward(chain: RealMatrix) : RealMatrix {

        val backpropagationActivation =  this.activationLayer.backward(chain)

        return this.statefulProjectionLayer.backward(backpropagationActivation)

    }

    override fun optimize() {

        this.statefulProjectionLayer.optimize()

    }

}
package shape.komputation.layers.continuation.recurrent

import shape.komputation.layers.continuation.ContinuationLayer
import shape.komputation.layers.continuation.OptimizableContinuationLayer
import shape.komputation.layers.continuation.activation.ActivationLayer
import shape.komputation.matrix.RealMatrix

class RecurrentLayer(
    name : String?,
    private val statefulProjectionLayer: StatefulProjectionLayer,
    private val activationLayer: ActivationLayer) : ContinuationLayer(name, 2, 3), OptimizableContinuationLayer {

    // activate(stateful_projection(input))
    override fun forward() {

        this.statefulProjectionLayer.setInput(this.lastInput!!)
        this.statefulProjectionLayer.forward()
        val projection = this.statefulProjectionLayer.lastForwardResult[0]

        this.activationLayer.setInput(projection)
        this.activationLayer.forward()
        val activation = this.activationLayer.lastForwardResult[0]

        this.lastForwardResult[0] = projection
        this.lastForwardResult[1] = activation

    }

    override fun backward(chain: RealMatrix) {

        this.activationLayer.backward(chain)

        this.statefulProjectionLayer.backward(this.activationLayer.lastBackwardResultWrtInput!!)

        this.lastBackwardResultWrtInput = this.statefulProjectionLayer.lastBackwardResultWrtInput!!

        this.lastBackwardResultWrtParameters[0] = this.statefulProjectionLayer.lastBackwardResultWrtParameters.first()
        this.lastBackwardResultWrtParameters[1] = this.statefulProjectionLayer.lastBackwardResultWrtParameters.last()

    }

    override fun optimize() {

        this.statefulProjectionLayer.optimize()

    }

}
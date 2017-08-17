package shape.komputation.cuda.layers.forward.activation

import jcuda.Pointer

class CudaIdentityLayer internal constructor(name : String? = null, numberRows : Int, numberColumns : Int) : BaseCudaActivationLayer(name) {

    override var deviceForwardResult = Pointer()
    override val numberInputRows = numberRows
    override val numberInputColumns = numberColumns

    override var deviceBackwardResult = Pointer()
    override val numberOutputRows = numberRows
    override val numberOutputColumns = numberColumns

    override fun forward(batchSize: Int, numberInputColumns : Int, input: Pointer, isTraining: Boolean): Pointer {

        this.deviceForwardResult = input

        return this.deviceForwardResult

    }

    override fun backward(batchSize: Int, chain: Pointer) : Pointer {

        this.deviceBackwardResult = chain

        return this.deviceBackwardResult

    }

}
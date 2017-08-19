package shape.komputation.cuda.layers

import jcuda.Pointer
import jcuda.runtime.JCuda.cudaFree
import shape.komputation.cuda.setIntArray
import shape.komputation.layers.Resourceful
import shape.komputation.matrix.constantIntArray

abstract class BaseCudaForwardLayer(private val name : String?) : CudaForwardLayer, Resourceful {

    override val deviceNumberInputColumns = Pointer()
    override val deviceNumberOutputColumns = Pointer()

    override fun acquire(maximumBatchSize: Int) {

        setIntArray(constantIntArray(maximumBatchSize, this.maximumInputColumns), maximumBatchSize, this.deviceNumberInputColumns)
        setIntArray(constantIntArray(maximumBatchSize, this.maximumOutputColumns), maximumBatchSize, this.deviceNumberOutputColumns)

    }

    override fun release() {

        cudaFree(this.deviceNumberInputColumns)
        cudaFree(this.deviceNumberOutputColumns)

    }

}
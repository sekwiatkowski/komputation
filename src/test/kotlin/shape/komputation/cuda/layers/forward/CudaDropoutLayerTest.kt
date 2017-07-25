package shape.komputation.cuda.layers.forward

import jcuda.Pointer
import jcuda.jcublas.cublasHandle
import org.junit.jupiter.api.Test
import shape.komputation.cuda.getVector
import shape.komputation.cuda.setUpCudaContext
import shape.komputation.layers.forward.dropout.dropoutLayer
import java.util.*

class CudaDropoutLayerTest {

    @Test
    fun test() {

        val cudaContext = setUpCudaContext()

        val layer = dropoutLayer(Random(1), 1, 0.5f).buildForCuda(cudaContext, cublasHandle())

        layer.acquire()

        val deviceResult = layer.forward(Pointer(), true)
        val vector = getVector(deviceResult, 1)

        layer.release()

        cudaContext.destroy()

    }

}
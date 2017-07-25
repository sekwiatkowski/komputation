package shape.komputation.cuda.layers.forward.dropout

import jcuda.Pointer
import shape.komputation.cuda.Kernel
import shape.komputation.cuda.allocateDeviceMemory
import shape.komputation.cuda.layers.forward.activation.BaseCudaActivationLayer
import shape.komputation.layers.Resourceful
import jcuda.runtime.JCuda.cudaFree

class CudaDropoutLayer internal constructor(
    name : String? = null,
    private val trainingKernel: Kernel) : BaseCudaActivationLayer(name), Resourceful {

    private val deviceResult = Pointer()
    private val pointerToDeviceResult = Pointer.to(this.deviceResult)

    override fun acquire() {

        /* val generator = curandGenerator()
        curandCreateGenerator(generator, CURAND_RNG_PSEUDO_DEFAULT)
        curandSetPseudoRandomGeneratorSeed(generator, 1234) */

        this.trainingKernel.acquire()

        allocateDeviceMemory(this.deviceResult, 1)

    }

    override fun forward(input : Pointer, isTraining : Boolean): Pointer {

        this.trainingKernel.launch(
            Pointer.to(
                this.pointerToDeviceResult
            ),
            1,
            1,
            0
        )

        return this.deviceResult

    }


    override fun backward(chain : Pointer) : Pointer {

        TODO()

    }

    override fun release() {

        this.trainingKernel.release()

        cudaFree(this.deviceResult)

    }

}
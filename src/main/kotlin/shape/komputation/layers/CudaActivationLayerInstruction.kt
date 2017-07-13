package shape.komputation.layers

import shape.komputation.cuda.CudaEnvironment
import shape.komputation.cuda.layers.forward.activation.CudaActivationLayer

interface CudaActivationLayerInstruction : CudaForwardLayerInstruction {

    override fun buildForCuda(environment : CudaEnvironment): CudaActivationLayer

}
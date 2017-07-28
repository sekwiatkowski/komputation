package shape.komputation.cuda.layers.forward.activation

import shape.komputation.cuda.Kernel

class CudaTanhLayer internal constructor(
    name : String? = null,
    createForwardKernel: () -> Kernel,
    createBackwardKernel: () -> Kernel,
    maximumThreadsPerBlock: Int,
    numberEntries : Int) : BaseCudaEntrywiseActivationLayer(name, createForwardKernel, createBackwardKernel, maximumThreadsPerBlock, numberEntries)
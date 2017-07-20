package shape.komputation.cuda.layers.forward.activation

import shape.komputation.cuda.Kernel

class CudaExponentiationLayer internal constructor(
    name : String? = null,
    forwardKernel: Kernel,
    backwardKernel : Kernel,
    maximumThreadsPerBlock : Int,
    numberEntries : Int) : BaseCudaEntrywiseActivationLayer(name, forwardKernel, backwardKernel, maximumThreadsPerBlock, numberEntries)
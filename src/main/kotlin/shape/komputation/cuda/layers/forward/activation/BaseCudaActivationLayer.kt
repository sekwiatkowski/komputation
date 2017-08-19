package shape.komputation.cuda.layers.forward.activation

import shape.komputation.cuda.layers.BaseCudaForwardLayer

abstract class BaseCudaActivationLayer(name : String?) : BaseCudaForwardLayer(name), CudaActivationLayer
package shape.komputation.cuda.optimization

import jcuda.Pointer
import shape.komputation.layers.Resourceful

interface CudaUpdateRule : Resourceful {

    fun update(pointerToDeviceParameter: Pointer, scalingFactor : Float, pointerToDeviceGradient : Pointer)

}
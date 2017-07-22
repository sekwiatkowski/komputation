package shape.komputation.cuda.optimization

import jcuda.Pointer

interface CudaUpdateRule {

    fun update(pointerToDeviceParameter: Pointer, scalingFactor : Float, pointerToDeviceGradient : Pointer)

}
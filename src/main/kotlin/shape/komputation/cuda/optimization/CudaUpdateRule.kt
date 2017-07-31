package shape.komputation.cuda.optimization

import jcuda.Pointer
import shape.komputation.layers.Resourceful

interface CudaUpdateRule : Resourceful {

    fun denseUpdate(pointerToParameters: Pointer, scalingFactor : Float, pointerToGradient: Pointer)

    fun sparseUpdate(numberParameters : Int, pointerToParameterIndices : Pointer, pointerToParameters: Pointer, scalingFactor : Float, pointerToGradient: Pointer)

}
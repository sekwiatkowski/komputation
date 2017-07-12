package shape.komputation.optimization

import jcuda.Pointer

interface CublasUpdateRule {

    fun update(deviceParameter: Pointer, scalingFactor : Double, deviceGradient : Pointer)

}
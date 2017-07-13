package shape.komputation.cuda.optimization

import jcuda.Pointer

interface CublasUpdateRule {

    fun update(deviceParameter: Pointer, scalingFactor : Double, deviceGradient : Pointer)

}
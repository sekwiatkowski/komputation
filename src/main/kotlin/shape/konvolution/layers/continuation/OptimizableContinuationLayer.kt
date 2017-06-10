package shape.konvolution.layers.continuation

import shape.konvolution.matrix.RealMatrix

interface OptimizableContinuationLayer {

    fun optimize(gradients : Array<RealMatrix?>)

}
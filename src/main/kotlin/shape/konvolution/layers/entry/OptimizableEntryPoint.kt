package shape.konvolution.layers.entry

import shape.konvolution.matrix.RealMatrix

interface OptimizableEntryPoint {

    fun optimize(chain : RealMatrix)

}
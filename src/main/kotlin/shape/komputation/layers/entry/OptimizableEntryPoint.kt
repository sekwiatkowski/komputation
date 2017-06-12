package shape.komputation.layers.entry

import shape.komputation.matrix.RealMatrix

interface OptimizableEntryPoint {

    fun optimize(chain : RealMatrix)

}
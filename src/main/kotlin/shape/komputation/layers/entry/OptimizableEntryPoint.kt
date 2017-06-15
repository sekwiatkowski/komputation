package shape.komputation.layers.entry

import shape.komputation.matrix.Matrix
import shape.komputation.matrix.RealMatrix

interface OptimizableEntryPoint {

    fun optimize(input : Matrix, chain : RealMatrix)

}
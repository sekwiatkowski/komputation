package shape.konvolution.layers.entry

import shape.konvolution.matrix.Matrix
import shape.konvolution.matrix.RealMatrix

interface OptimizableEntryPoint {

    fun optimize(input : Matrix, output: Array<RealMatrix>, gradient : RealMatrix)

}
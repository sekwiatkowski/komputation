package shape.konvolution

import shape.konvolution.matrix.RealMatrix

data class BackwardResult (
    val input : RealMatrix,
    val parameter : Array<RealMatrix?>? = null
)
package shape.konvolution

data class BackwardResult (
    val input : RealMatrix,
    val parameter : Array<RealMatrix?>? = null
)
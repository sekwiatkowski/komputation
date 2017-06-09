package shape.konvolution

import no.uib.cipr.matrix.Matrix

data class BackwardResult (
    val input : Matrix,
    val parameter : Array<Matrix?>? = null
)
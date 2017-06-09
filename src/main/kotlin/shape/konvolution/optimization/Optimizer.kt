package shape.konvolution.optimization

import no.uib.cipr.matrix.Matrix

interface Optimizer {

    fun optimize(parameter: Matrix, gradient: Matrix)

}
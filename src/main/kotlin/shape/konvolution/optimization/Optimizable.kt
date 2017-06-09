package shape.konvolution.optimization

import no.uib.cipr.matrix.Matrix

interface Optimizable {

    fun optimize(gradients : Array<Matrix?>)

}
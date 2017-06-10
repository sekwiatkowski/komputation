package shape.konvolution.optimization

import shape.konvolution.matrix.RealMatrix

interface Optimizer {

    fun optimize(parameter: RealMatrix, gradient: RealMatrix)

}
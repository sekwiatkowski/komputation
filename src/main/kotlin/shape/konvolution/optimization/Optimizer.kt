package shape.konvolution.optimization

import shape.konvolution.RealMatrix

interface Optimizer {

    fun optimize(parameter: RealMatrix, gradient: RealMatrix)

}
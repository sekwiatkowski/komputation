package shape.konvolution.optimization

import shape.konvolution.RealMatrix

interface Optimizable {

    fun optimize(gradients : Array<RealMatrix?>)

}
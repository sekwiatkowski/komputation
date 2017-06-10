package shape.konvolution.optimization

import shape.konvolution.matrix.RealMatrix

class StochasticGradientDescent(private val learningRate : Double) : Optimizer {

    override fun optimize(parameter: RealMatrix, gradient: RealMatrix) {

        for (indexRow in 0..parameter.numberRows() - 1) {

            for (indexColumn in 0..parameter.numberColumns() - 1) {

                val current = parameter.get(indexRow, indexColumn)
                val updated = current - learningRate * gradient.get(indexRow, indexColumn)

                parameter.set(indexRow, indexColumn, updated)

            }

        }

    }

}
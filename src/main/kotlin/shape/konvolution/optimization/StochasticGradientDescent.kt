package shape.konvolution.optimization

import no.uib.cipr.matrix.Matrix

class StochasticGradientDescent(private val learningRate : Double) : Optimizer {

    override fun optimize(parameter: Matrix, gradient: Matrix) {

        for (indexRow in 0..parameter.numRows() - 1) {

            for (indexColumn in 0..parameter.numColumns() - 1) {

                val current = parameter.get(indexRow, indexColumn)
                val updated = current - learningRate * gradient.get(indexRow, indexColumn)

                parameter.set(indexRow, indexColumn, updated)

            }

        }

    }

}
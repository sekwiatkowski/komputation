package shape.komputation.layers.forward.units

import shape.komputation.matrix.DoubleMatrix

abstract class RecurrentUnit(val name : String?) {

    abstract fun forwardStep(step : Int, state: DoubleMatrix, input: DoubleMatrix, isTraining : Boolean) : DoubleMatrix

    abstract fun backwardStep(step : Int, chain : DoubleMatrix): Pair<DoubleMatrix, DoubleMatrix>

    abstract fun backwardSeries()

}
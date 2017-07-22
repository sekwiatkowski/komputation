package shape.komputation.cpu.layers.forward.units

import shape.komputation.matrix.FloatMatrix

abstract class RecurrentUnit (val name : String?) {

    abstract fun forwardStep(step : Int, state: FloatMatrix, input: FloatMatrix, isTraining : Boolean) : FloatMatrix

    abstract fun backwardStep(step : Int, chain : FloatMatrix): Pair<FloatMatrix, FloatMatrix>

    abstract fun backwardSeries()

}
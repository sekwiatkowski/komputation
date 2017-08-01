package shape.komputation.cpu.layers.forward.units

import shape.komputation.matrix.FloatMatrix

abstract class RecurrentUnit (val name : String?) {

    abstract fun forwardStep(withinBatch : Int, step : Int, state: FloatMatrix, input: FloatMatrix, isTraining : Boolean) : FloatMatrix

    abstract fun backwardStep(withinBatch : Int, step : Int, chain : FloatMatrix): Pair<FloatMatrix, FloatMatrix>

    abstract fun backwardSeries()

}
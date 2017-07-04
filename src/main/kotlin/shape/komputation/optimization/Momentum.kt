package shape.komputation.optimization

fun momentum(learningRate: Double, momentum: Double): (Int, Int) -> UpdateRule {

    return { numberRows : Int, numberColumns : Int ->

        Momentum(learningRate, momentum, numberRows, numberColumns)

    }
}

class Momentum(private val learningRate: Double, private val momentum: Double, numberRows : Int, numberColumns : Int) : UpdateRule {

    private val history = DoubleArray(numberRows * numberColumns)

    override fun apply(index : Int, current: Double, derivative: Double): Double {

        val newStateEntry = this.momentum * this.history[index] + this.learningRate * derivative

        this.history[index] = newStateEntry

        val result = current - newStateEntry

        return result


    }

}
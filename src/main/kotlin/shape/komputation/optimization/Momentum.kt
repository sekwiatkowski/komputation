package shape.komputation.optimization

fun momentum(learningRate: Double, momentum: Double): (Int, Int) -> UpdateRule {

    return { numberRows : Int, numberColumns : Int ->

        Momentum(learningRate, momentum, numberRows, numberColumns)

    }
}

class Momentum(private val learningRate: Double, private val momentum: Double, val numberRows : Int, val numberColumns : Int) : UpdateRule {

    val state = DoubleArray(numberRows * numberColumns)

    override fun apply(index : Int, current: Double, derivative: Double): Double {

        val newStateEntry = momentum * state[index] + learningRate * derivative

        state[index] = newStateEntry

        val result = current - newStateEntry

        return result


    }

}
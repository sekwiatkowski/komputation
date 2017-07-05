package shape.komputation.optimization

fun momentum(learningRate: Double, momentum: Double): (Int, Int) -> UpdateRule {

    return { numberRows : Int, numberColumns : Int ->

        Momentum(learningRate, momentum, numberRows, numberColumns)

    }
}

/*
    After n step, the first scaled gradient is decayed by momentum^(n-1), the second scaled gradient is decayed by momentum^(n-2), etc.
    history_0 = 0
    history_1 = momentum * history_0 - learning_rate * gradient_1
              = - learning_rate * gradient_1
    history_2 = momentum * history_1 - learning_rate * gradient_2
              = momentum * (- learning_rate * gradient_1) - learning_rate * gradient_2
              = - momentum * learning_rate * gradient_1 - learning_rate * gradient_2
    history_3 = momentum * history_2 - learning_rate * gradient_3
              = momentum * (- momentum * learning_rate * gradient_1 - learning_rate * gradient_2) - learning_rate * gradient_3
              = - momentum^2 * learning_rate * gradient_1 - momentum * learning_rate * gradient 2 - learning_rate * gradient_3
 */
class Momentum(private val learningRate: Double, private val momentum: Double, numberRows : Int, numberColumns : Int) : UpdateRule {

    private val history = DoubleArray(numberRows * numberColumns)

    override fun apply(index : Int, current: Double, derivative: Double): Double {

        val newStateEntry = this.momentum * this.history[index] - this.learningRate * derivative

        this.history[index] = newStateEntry

        val result = current + newStateEntry

        return result

    }

}
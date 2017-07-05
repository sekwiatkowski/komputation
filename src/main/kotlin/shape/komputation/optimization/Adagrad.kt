package shape.komputation.optimization

fun adagrad(learningRate: Double, epsilon: Double = 1e-6): (Int, Int) -> UpdateRule {

    return { numberRows : Int, numberColumns : Int ->

        Adagrad(learningRate, epsilon, numberRows, numberColumns)

    }

}

class Adagrad(private val learningRate: Double, private val epsilon : Double, numberRows : Int, val numberColumns : Int) : UpdateRule {

    private val sumOfSquaredDerivatives = DoubleArray(numberRows * numberColumns)

    override fun apply(index : Int, current: Double, derivative: Double): Double {

        this.sumOfSquaredDerivatives[index] += Math.pow(derivative, 2.0)

        val adaptiveLearningRate = this.learningRate / (this.epsilon + Math.sqrt(this.sumOfSquaredDerivatives[index]))

        val update = - adaptiveLearningRate * derivative

        return current + update
    }

}
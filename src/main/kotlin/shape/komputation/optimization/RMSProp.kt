package shape.komputation.optimization

fun rmsprop(learningRate: Double, decay : Double = 0.9, epsilon: Double = 1e-6): (Int, Int) -> UpdateRule {

    return { numberRows : Int, numberColumns : Int ->

        RMSProp(learningRate, decay, epsilon, numberRows, numberColumns)

    }

}

class RMSProp(private val learningRate : Double, private val decay : Double, private val epsilon : Double, numberRows : Int, val numberColumns : Int) : UpdateRule {

    private val oneMinusDecay = 1.0 - this.decay

    private val accumulation = DoubleArray(numberRows * numberColumns)

    override fun apply(index : Int, current: Double, derivative: Double): Double {

        val newAccumulation = this.decay * this.accumulation[index] + this.oneMinusDecay * Math.pow(derivative, 2.0)
        this.accumulation[index] = newAccumulation

        val adaptiveLearningRate = this.learningRate / Math.sqrt(newAccumulation + this.epsilon)

        val update = - adaptiveLearningRate * derivative

        return current + update

    }

}
package shape.komputation.optimization

fun adadelta(decay : Double = 0.95, epsilon: Double = 1e-6): (Int, Int) -> UpdateRule {

    return { numberRows : Int, numberColumns : Int ->

        Adadelta(decay, epsilon, numberRows, numberColumns)

    }

}

class Adadelta(private val decay : Double, private val epsilon : Double, numberRows : Int, val numberColumns : Int) : UpdateRule {

    private val oneMinusDecay = 1.0 - this.decay

    private val gradientAccumulation = DoubleArray(numberRows * numberColumns)
    private val updateAccumulation = DoubleArray(numberRows * numberColumns)

    override fun apply(index : Int, current: Double, derivative: Double): Double {

        val newGradientAccumulation = this.decay * this.gradientAccumulation[index] + this.oneMinusDecay * Math.pow(derivative, 2.0)
        this.gradientAccumulation[index] = newGradientAccumulation
        val rootMeanSquaredOfDerivatives = Math.sqrt(newGradientAccumulation + this.epsilon)

        val pastUpdateAccumulation = this.updateAccumulation[index]
        val rootMeanSquaredOfPastUpdates = Math.sqrt(pastUpdateAccumulation + this.epsilon)

        val learningRate = rootMeanSquaredOfPastUpdates / rootMeanSquaredOfDerivatives

        val update = - learningRate * derivative

        this.updateAccumulation[index] = this.decay * pastUpdateAccumulation + this.oneMinusDecay * Math.pow(update, 2.0)

        val result = current + update

        return result

    }

}
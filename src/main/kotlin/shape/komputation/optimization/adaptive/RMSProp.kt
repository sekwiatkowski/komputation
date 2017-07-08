package shape.komputation.optimization.adaptive

import shape.komputation.optimization.UpdateRule

fun rmsprop(learningRate: Double, decay : Double = 0.9, epsilon: Double = 1e-6): (Int, Int) -> UpdateRule {

    return { numberRows : Int, numberColumns : Int ->

        RMSProp(learningRate, decay, epsilon, numberRows * numberColumns)

    }

}

class RMSProp(private val learningRate : Double, private val decay : Double, private val epsilon : Double, size : Int) : UpdateRule {

    private val oneMinusDecay = 1.0 - this.decay

    private val accumulation = DoubleArray(size)

    override fun updateSparsely(start : Int, parameters: DoubleArray, gradient: DoubleArray, gradientSize : Int) {

        for(index in 0..gradientSize-1) {

            val derivative = gradient[index]

            val historyIndex = start + index

            val newAccumulation = this.decay * this.accumulation[historyIndex] + this.oneMinusDecay * Math.pow(derivative, 2.0)
            this.accumulation[historyIndex] = newAccumulation

            val adaptiveLearningRate = this.learningRate / Math.sqrt(newAccumulation + this.epsilon)

            val update = -adaptiveLearningRate * derivative

            parameters[index] += update

        }

    }

}
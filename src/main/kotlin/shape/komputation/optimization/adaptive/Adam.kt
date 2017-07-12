package shape.komputation.optimization.adaptive

import shape.komputation.optimization.UpdateRule

fun adam(learningRate : Double = 0.001, firstMomentDecay : Double = 0.9, secondMomentDecay : Double = 0.999, epsilon : Double = 1e-8): (Int, Int) -> UpdateRule {

    return { numberRows : Int, numberColumns : Int ->

        Adam(learningRate, firstMomentDecay, secondMomentDecay, epsilon, numberRows * numberColumns)

    }

}

class Adam(
    private val learningRate : Double,
    private val firstMomentDecay : Double,
    private val secondMomentDecay : Double,
    private val epsilon : Double, size : Int) : UpdateRule {

    private val oneMinusFirstMomentDecay = 1.0 - firstMomentDecay
    private val oneMinusSecondMomentDecay = 1.0 - secondMomentDecay

    private val firstMomentEstimate = DoubleArray(size)
    private val secondMomentEstimate = DoubleArray(size)

    private var step = 0.0

    override fun updateSparsely(start : Int, parameters: DoubleArray, gradient: DoubleArray, gradientSize : Int) {

        this.step += 1.0

        for (index in 0..gradientSize - 1) {

            val derivative = gradient[index]

            val updatedFirstMomentEstimate = this.firstMomentDecay * this.firstMomentEstimate[index] + this.oneMinusFirstMomentDecay * derivative
            this.firstMomentEstimate[index] = updatedFirstMomentEstimate
            val correctedFirstMomentEstimate = updatedFirstMomentEstimate / (1.0 - Math.pow(this.firstMomentDecay, this.step))

            val updatedSecondMomentEstimate = this.secondMomentDecay * this.secondMomentEstimate[index] + this.oneMinusSecondMomentDecay * derivative * derivative
            this.secondMomentEstimate[index] = updatedSecondMomentEstimate
            val correctedSecondMomentEstimate = updatedSecondMomentEstimate / (1.0 - Math.pow(this.secondMomentDecay, this.step))

            val adaptedLearningRate = this.learningRate / (Math.sqrt(correctedSecondMomentEstimate) + this.epsilon)

            val change = -correctedFirstMomentEstimate * adaptedLearningRate

            parameters[index] += change

        }

    }

}
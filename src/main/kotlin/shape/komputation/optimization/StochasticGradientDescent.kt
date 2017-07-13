package shape.komputation.optimization

fun stochasticGradientDescent(learningRate: Double): (Int, Int) -> UpdateRule {

    return { _: Int, _: Int ->

        StochasticGradientDescent(learningRate)

    }

}

class StochasticGradientDescent(private val learningRate: Double) : UpdateRule {

    override fun updateSparsely(start : Int, parameters: DoubleArray, gradient: DoubleArray, gradientSize : Int) {

        for(index in 0..gradientSize-1) {

            parameters[index] -= learningRate * gradient[index]

        }

    }

}
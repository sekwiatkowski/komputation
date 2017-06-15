package shape.komputation.optimization

fun stochasticGradientDescent(learningRate: Double): (Int, Int) -> UpdateRule {

    return { numberRows : Int, numberColumns : Int ->

        StochasticGradientDescent(learningRate)

    }

}

class StochasticGradientDescent(private val learningRate: Double) : UpdateRule {

    override fun apply(index : Int, current: Double, derivative: Double): Double {

        return current - learningRate * derivative

    }

}
package shape.konvolution.optimization

fun stochasticGradientDescent(learningRate: Double): (Int, Int) -> UpdateRule {

    return { numberRows : Int, numberColumns : Int ->

        val updateRule = { indexRow: Int, indexColumn: Int, current: Double, derivative: Double ->

            current - learningRate * derivative

        }

        updateRule

    }

}
package shape.komputation.optimization

import shape.komputation.functions.scale

fun updateDensely(parameters: DoubleArray, gradient : DoubleArray, scalingFactor : Double, rule : UpdateRule) {

    val scaledGradient = scale(gradient, scalingFactor)

    rule.updateDensely(parameters, scaledGradient, scaledGradient.size)

}
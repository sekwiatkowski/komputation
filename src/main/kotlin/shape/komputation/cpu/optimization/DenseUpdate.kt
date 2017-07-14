package shape.komputation.cpu.optimization

import shape.komputation.cpu.functions.scale

fun updateDensely(parameters: DoubleArray, gradient : DoubleArray, scalingFactor : Double, rule : UpdateRule) {

    val scaledGradient = scale(gradient, scalingFactor)

    rule.updateDensely(parameters, scaledGradient, scaledGradient.size)

}
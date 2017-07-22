package shape.komputation.cpu.optimization

import shape.komputation.cpu.functions.scale

fun updateDensely(parameters: FloatArray, gradient : FloatArray, scalingFactor : Float, rule : UpdateRule) {

    val scaledGradient = scale(gradient, scalingFactor)

    rule.updateDensely(parameters, scaledGradient, scaledGradient.size)

}
package shape.komputation.cpu.optimization

import shape.komputation.cpu.functions.scale

fun updateDensely(parameters: FloatArray, gradient : FloatArray, dimension : Int, scalingFactor : Float, rule : UpdateRule) {

    scale(gradient, scalingFactor, dimension)

    rule.updateDensely(parameters, gradient, dimension)

}
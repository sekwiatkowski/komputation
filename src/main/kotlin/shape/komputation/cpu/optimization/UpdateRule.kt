package shape.komputation.cpu.optimization

interface UpdateRule {

    fun updateDensely(parameters : DoubleArray, gradient : DoubleArray, gradientSize : Int) =

        updateSparsely(0, parameters, gradient, gradientSize)

    fun updateSparsely(start : Int, parameters : DoubleArray, gradient : DoubleArray, gradientSize : Int)

}
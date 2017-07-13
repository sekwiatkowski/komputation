package shape.komputation.cpu.optimization

class CpuStochasticGradientDescent(private val learningRate: Double) : UpdateRule {

    override fun updateSparsely(start : Int, parameters: DoubleArray, gradient: DoubleArray, gradientSize : Int) {

        for(index in 0..gradientSize-1) {

            parameters[index] -= learningRate * gradient[index]

        }

    }

}
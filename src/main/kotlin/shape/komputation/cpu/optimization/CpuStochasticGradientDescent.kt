package shape.komputation.cpu.optimization

class CpuStochasticGradientDescent(private val learningRate: Float) : UpdateRule {

    override fun updateSparsely(start : Int, parameters: FloatArray, gradient: FloatArray, gradientSize : Int) {

        for(index in 0..gradientSize-1) {

            parameters[index] -= this.learningRate * gradient[index]

        }

    }

}
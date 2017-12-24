package com.komputation.cpu.optimization

class CpuStochasticGradientDescent(private val learningRate: Float) : UpdateRule {

    override fun updateSparsely(start : Int, parameter: FloatArray, gradient: FloatArray, dimension: Int) {
        for(index in 0 until dimension) {
            parameter[index] -= this.learningRate * gradient[index]
        }
    }

}
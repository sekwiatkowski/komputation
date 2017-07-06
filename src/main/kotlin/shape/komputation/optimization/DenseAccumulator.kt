package shape.komputation.optimization

import java.util.*

class DenseAccumulator(private val size : Int) {

    private var accumulation = DoubleArray(size)

    fun accumulate(gradient: DoubleArray) {

        for (index in 0..this.size - 1) {

            accumulation[index] += gradient[index]

        }

    }

    fun getAccumulation() = this.accumulation

    fun reset() {

        Arrays.fill(accumulation, 0.0)

    }

}



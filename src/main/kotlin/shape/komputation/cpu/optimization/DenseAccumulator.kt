package shape.komputation.cpu.optimization

import java.util.*

class DenseAccumulator(private val size : Int) {

    private var accumulation = FloatArray(size)

    fun accumulate(gradient: FloatArray) {

        for (index in 0..this.size - 1) {

            accumulation[index] += gradient[index]

        }

    }

    fun getAccumulation() = this.accumulation

    fun reset() {

        Arrays.fill(accumulation, 0.0f)

    }

}



package shape.komputation.optimization

import java.util.*

class DenseAccumulator(private val size : Int) {

    private var accumulation = DoubleArray(size)

    private var count = 0

    fun accumulate(gradient: DoubleArray) {

        for (index in 0..this.size - 1) {

            accumulation[index] += gradient[index]

        }

        this.count++

    }

    fun getAccumulation() = this.accumulation

    fun getCount() = this.count

    fun reset() {

        this.count = 0

        Arrays.fill(accumulation, 0.0)

    }

}



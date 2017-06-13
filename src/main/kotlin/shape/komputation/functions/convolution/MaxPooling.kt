package shape.komputation.functions.convolution

import shape.komputation.matrix.RealMatrix
import shape.komputation.matrix.createRealMatrix

fun maxPooling(input: RealMatrix): RealMatrix {

    val maxPooled = createRealMatrix(input.numberRows(), 1)

    var index = 0

    for (indexRow in 0..input.numberRows() - 1) {

        var maxValue = Double.NEGATIVE_INFINITY

        for (indexColumn in 0..input.numberColumns() - 1) {

            val entry = input.get(indexRow, indexColumn)

            if (entry > maxValue) {
                maxValue = entry
            }

        }

        maxPooled.set(index++, 0, maxValue)

    }

    return maxPooled

}

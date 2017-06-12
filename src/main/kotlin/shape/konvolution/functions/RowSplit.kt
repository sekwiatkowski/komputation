package shape.konvolution.functions

import shape.konvolution.matrix.RealMatrix
import shape.konvolution.matrix.createRealMatrix

fun splitRows(matrix : RealMatrix, heights : IntArray): Array<RealMatrix> {

    val numberColumns = matrix.numberColumns()

    var indexRow = 0

    return Array<RealMatrix>(heights.size) { index ->

        val height = heights[index]

        val subMatrix = createRealMatrix(height, numberColumns)

        for (indexSubRow in 0..height - 1) {

            for (indexColumn in 0..numberColumns - 1) {

                subMatrix.set(indexSubRow, indexColumn, matrix.get(indexRow, indexColumn))

            }

            indexRow += 1

        }

        subMatrix

    }

}
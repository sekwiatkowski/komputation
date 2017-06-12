package shape.konvolution.functions

import shape.konvolution.matrix.RealMatrix
import shape.konvolution.matrix.createRealMatrix

fun softmax(input: RealMatrix) =

    createRealMatrix(input.numberRows(), input.numberColumns()).let { activated ->

        for (indexColumn in 0..input.numberColumns() - 1) {

            val exponentiated = Array(input.numberRows()) { indexRow ->

                Math.exp(input.get(indexRow, indexColumn))

            }

            val sum = exponentiated.sum()

            for (indexRow in 0..input.numberRows() - 1) {

                val activation = exponentiated[indexRow].div(sum)

                activated.set(indexRow, indexColumn, activation)
            }


        }

        activated

    }
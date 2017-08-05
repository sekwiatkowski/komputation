package shape.komputation.cpu.functions

import java.util.*

/*
    word^(1)_1   word^(2)_1   ...   word^(T)_1
    word^(1)_2   word^(2)_2   ...   word^(T)_2
    ...          ...                ....
    word^(1)_d   word^(2)_d   ...   word^(T)_d
*/


fun lookup(vectors : Array<FloatArray>, length : Int, dimension : Int, padding : Float, ids : IntArray, result : FloatArray) {

    val numberIds = ids.size

    for (index in 0..numberIds - 1) {

        val id = ids[index]

        val start = index * dimension

        val vector = vectors[id]

        System.arraycopy(vector, 0, result, start, dimension)

    }

    val remainingLength = length - numberIds

    if(remainingLength > 0) {

        Arrays.fill(result, numberIds * dimension, length * dimension, padding)

    }

}
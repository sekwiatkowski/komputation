package shape.komputation.cpu.functions

import shape.komputation.matrix.xorShift
import java.util.*

fun seed(random : Random, result: IntArray, numberEntries: Int) {

    for (index in 0..numberEntries - 1) {

        result[index] = random.nextInt()

    }

}

fun mask(seeds : IntArray, limit : Int, result : BooleanArray, numberEntries : Int) {

    for (index in 0..numberEntries-1) {

        val updatedSeed = xorShift(seeds[index])

        seeds[index] = updatedSeed

        result[index] = updatedSeed > limit

    }

}

fun dropout(input : FloatArray, mask : BooleanArray, result : FloatArray, numberEntries: Int) {

    for (index in 0..numberEntries - 1) {

        result[index] = if(mask[index]) input[index] else 0.0f

    }

}

fun backwardDropout(chain : FloatArray, mask : BooleanArray, result: FloatArray, numberEntries: Int) {

    for (index in 0..numberEntries - 1) {

        result[index] = if(mask[index]) chain[index] else 0.0f

    }

}
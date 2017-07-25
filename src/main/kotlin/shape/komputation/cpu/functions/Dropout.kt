package shape.komputation.cpu.functions

import java.util.*

fun mask(random : Random, keepProbability: Float, result : BooleanArray, numberEntries : Int) {

    for (index in 0..numberEntries - 1) {

        result[index] = random.nextFloat() <= keepProbability

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
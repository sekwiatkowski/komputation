package shape.komputation.cpu.functions

fun transpose(numberRows: Int, numberColumns: Int, entries: FloatArray): FloatArray {

    if(numberRows == 1 || numberColumns == 1) {

        return entries

    }

    val result = FloatArray(entries.size)

    for (indexColumn in 0..numberColumns-1) {

        val start = indexColumn * numberRows

        for(indexRow in 0..numberRows-1) {

            result[indexRow * numberColumns + indexColumn] = entries[start + indexRow]

        }

    }

    return result

}
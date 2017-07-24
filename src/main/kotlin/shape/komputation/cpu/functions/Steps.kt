package shape.komputation.cpu.functions

fun getStep(entries : FloatArray, indexStep: Int, result : FloatArray, numberRows : Int) {

    System.arraycopy(entries, indexStep * numberRows, result, 0, numberRows)

}

fun setStep(entries: FloatArray, indexStep: Int, step : FloatArray, numberRows : Int) {

    System.arraycopy(step, 0, entries, indexStep * numberRows, numberRows)

}
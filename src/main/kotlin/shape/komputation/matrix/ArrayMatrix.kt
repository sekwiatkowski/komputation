package shape.komputation.matrix

sealed class Matrix(val numberRows: Int, val numberColumns: Int)

open class FloatMatrix(numberRows: Int, numberColumns: Int, val entries: FloatArray) : Matrix(numberRows, numberColumns)

class IntMatrix(numberRows : Int, numberColumns : Int, val entries : IntArray) : Matrix(numberRows, numberColumns)

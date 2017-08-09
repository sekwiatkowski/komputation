package shape.komputation.matrix

sealed class Matrix()

open class FloatMatrix(val entries: FloatArray) : Matrix()

class IntMatrix(val entries : IntArray) : Matrix()

package shape.komputation.initialization

class ConstantInitialization internal constructor(private val constant: Double) : InitializationStrategy {

    override fun initialize(indexRow: Int, indexColumn: Int, numberIncoming: Int)  =

        constant
}

fun constantInitialization(constant: Double) = ConstantInitialization(constant)
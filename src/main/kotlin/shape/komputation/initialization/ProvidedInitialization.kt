package shape.komputation.initialization

class ProvidedInitialization internal constructor(private val entries: DoubleArray, private val numberRows : Int) : InitializationStrategy {

    override fun initialize(indexRow: Int, indexColumn: Int, numberIncoming: Int)  =

        this.entries[indexRow + indexColumn * this.numberRows]
}

fun providedInitialization(entries : DoubleArray, numberRows: Int) = ProvidedInitialization(entries, numberRows)
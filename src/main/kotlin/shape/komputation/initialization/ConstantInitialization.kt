package shape.komputation.initialization

fun constantInitialization(constant: Double) : InitializationStrategy {

    return { _, _ -> constant }

}
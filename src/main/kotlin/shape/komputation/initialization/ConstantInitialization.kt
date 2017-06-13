package shape.komputation.initialization

fun createConstantInitializer(constant: Double): InitializationStrategy {

    return { _, _ -> constant }

}

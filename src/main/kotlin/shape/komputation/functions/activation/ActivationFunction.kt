package shape.komputation.functions.activation

enum class ActivationFunction(val layerName : String) {

    Identity("identity"),
    ReLU("relu"),
    Sigmoid("sigmoid"),
    Softmax("softmax"),
    Tanh("tanh")

}
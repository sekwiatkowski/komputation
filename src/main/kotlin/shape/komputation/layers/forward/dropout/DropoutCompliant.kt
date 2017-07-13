package shape.komputation.layers.forward.dropout

import shape.komputation.layers.ForwardLayer
import shape.komputation.layers.SparseForwarding

interface DropoutCompliant : ForwardLayer, SparseForwarding
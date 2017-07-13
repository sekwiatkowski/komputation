package shape.komputation.cpu.layers.forward.dropout

import shape.komputation.cpu.layers.SparseForwarding
import shape.komputation.cpu.layers.forward.activation.CpuActivationLayer

interface DropoutCompliant : CpuActivationLayer, SparseForwarding
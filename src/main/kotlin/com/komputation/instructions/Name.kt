package com.komputation.instructions

fun concatenateNames(baseName: String?, appendix: String) =
    if(baseName != null)
        "$baseName-$appendix"
    else
        null
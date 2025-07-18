#pragma once

enum class OptionType : int { EUROPEAN = 0, ASIAN = 1, BARRIER = 2 };
enum class BarrierType : int { NONE = 0, UP_AND_OUT = 1, DOWN_AND_OUT = 2 };
enum class OptionStyle : int { CALL = 0, PUT = 1 };

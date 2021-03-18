#include "utils.h"

#include <sstream>
#include <cmath>

std::string ToStringPercentage(float v) {
    float value = round(v * 10000.0f) / 100.0f;
    std::ostringstream out;
    out.precision(2);
    out << std::fixed << value;
    return out.str();
}

std::string ToString(const std::vector<int> & v) {
    std::string buf = "{";
    const int len = v.size();
    for (int i = 0; i < (len - 1); i++) {
        buf += std::to_string(v[i]) + ",";
    }
    if (len > 0) {
        buf += std::to_string(v[len - 1]);
    }
    buf += "}";

    return buf;
}
#pragma once

#include <cstdarg>

namespace rtr::core {

enum class LogLevel { Info, Warning, Error };

class Logger {
public:
    static void info(const char* tag, const char* format, ...);
    static void warn(const char* tag, const char* format, ...);
    static void error(const char* tag, const char* format, ...);

private:
    static void log(LogLevel level, const char* tag, const char* format, std::va_list args);
};

}  // namespace rtr::core

#include "RTRMetalEngine/Core/Logger.hpp"

#include <cstdarg>
#include <cstdio>
#include <mutex>
#include <string>
#include <vector>

namespace rtr::core {
namespace {

std::mutex& logMutex() {
    static std::mutex mutex;
    return mutex;
}

LogLevel& minLogLevel() {
    static LogLevel level = LogLevel::Info;
    return level;
}

const char* levelLabel(LogLevel level) {
    switch (level) {
    case LogLevel::Info:
        return "INFO";
    case LogLevel::Warning:
        return "WARN";
    case LogLevel::Error:
        return "ERROR";
    }
    return "INFO";
}

std::string formatMessage(const char* format, std::va_list args) {
    std::va_list argsCopy;
    va_copy(argsCopy, args);
    const int length = std::vsnprintf(nullptr, 0, format, argsCopy);
    va_end(argsCopy);

    if (length <= 0) {
        return {};
    }

    std::vector<char> buffer(static_cast<std::size_t>(length) + 1);
    std::vsnprintf(buffer.data(), buffer.size(), format, args);
    return std::string(buffer.data());
}

}  // namespace

void Logger::log(LogLevel level, const char* tag, const char* format, std::va_list args) {
    if (static_cast<int>(level) < static_cast<int>(minLogLevel())) {
        return;
    }
    const std::string message = formatMessage(format, args);
    std::lock_guard<std::mutex> lock(logMutex());
    std::fprintf(stderr, "[%s][%s] %s\n", levelLabel(level), tag ? tag : "RTR", message.c_str());
}

void Logger::info(const char* tag, const char* format, ...) {
    std::va_list args;
    va_start(args, format);
    log(LogLevel::Info, tag, format, args);
    va_end(args);
}

void Logger::warn(const char* tag, const char* format, ...) {
    std::va_list args;
    va_start(args, format);
    log(LogLevel::Warning, tag, format, args);
    va_end(args);
}

void Logger::error(const char* tag, const char* format, ...) {
    std::va_list args;
    va_start(args, format);
    log(LogLevel::Error, tag, format, args);
    va_end(args);
}

void Logger::setMinimumLevel(LogLevel level) { minLogLevel() = level; }

LogLevel Logger::minimumLevel() { return minLogLevel(); }

}  // namespace rtr::core

#pragma once
// =============================================================================
// logger.hpp — Thread-safe logger for CUDA tutorials
//
// Reads log level and log file name from YAML config.
// Writes to both stdout and a file under the logs/ directory.
//
// Usage:
//   auto logger = Logger::create("configs/tutorials/01_foo.yaml", PROJECT_ROOT);
//   logger.log_info("hello world");
// =============================================================================

#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include <yaml-cpp/yaml.h>

namespace cuda_tutorials {

// ---------------------------------------------------------------------------
// LogLevel enum — ordered so numeric comparison works for filtering
// ---------------------------------------------------------------------------
enum class LogLevel : int {
    DEBUG   = 0,
    INFO    = 1,
    WARNING = 2,
    ERROR   = 3
};

inline LogLevel log_level_from_string(const std::string& s) {
    static const std::unordered_map<std::string, LogLevel> table = {
        {"DEBUG",   LogLevel::DEBUG},
        {"INFO",    LogLevel::INFO},
        {"WARNING", LogLevel::WARNING},
        {"WARN",    LogLevel::WARNING},
        {"ERROR",   LogLevel::ERROR},
    };
    auto it = table.find(s);
    if (it == table.end()) {
        throw std::invalid_argument("Unknown log level: " + s);
    }
    return it->second;
}

inline std::string log_level_to_string(LogLevel level) {
    switch (level) {
        case LogLevel::DEBUG:   return "DEBUG";
        case LogLevel::INFO:    return "INFO";
        case LogLevel::WARNING: return "WARNING";
        case LogLevel::ERROR:   return "ERROR";
    }
    return "UNKNOWN";
}

// ---------------------------------------------------------------------------
// Logger — thread-safe, writes to stdout + log file
// ---------------------------------------------------------------------------
class Logger {
public:
    // Factory: reads logging.level and logging.file from YAML config.
    // project_root: path to the project root (logs/ folder lives there).
    // tutorial_name: used as the [source] field in each log line.
    static Logger create(const std::string& config_path,
                         const std::string& project_root,
                         const std::string& tutorial_name) {
        YAML::Node cfg = YAML::LoadFile(config_path);

        std::string level_str = "INFO";
        std::string log_file  = "tutorial.log";

        if (cfg["logging"]) {
            if (cfg["logging"]["level"]) {
                level_str = cfg["logging"]["level"].as<std::string>();
            }
            if (cfg["logging"]["file"]) {
                log_file = cfg["logging"]["file"].as<std::string>();
            }
        }

        LogLevel level = log_level_from_string(level_str);

        // Construct full path: <project_root>/logs/<log_file>
        std::filesystem::path logs_dir =
            std::filesystem::path(project_root) / "logs";
        std::filesystem::create_directories(logs_dir);
        std::filesystem::path full_log_path = logs_dir / log_file;

        return Logger(level, full_log_path.string(), tutorial_name);
    }

    // Manual construction (if you want to bypass YAML)
    Logger(LogLevel level,
           const std::string& log_file_path,
           const std::string& tutorial_name)
        : min_level_(level),
          tutorial_name_(tutorial_name),
          file_stream_(log_file_path, std::ios::app) {
        if (!file_stream_.is_open()) {
            throw std::runtime_error("Cannot open log file: " + log_file_path);
        }
    }

    // Non-copyable, movable
    Logger(const Logger&)            = delete;
    Logger& operator=(const Logger&) = delete;
    Logger(Logger&&)                 = default;
    Logger& operator=(Logger&&)      = default;

    void log_debug(const std::string& msg)   { write(LogLevel::DEBUG,   msg); }
    void log_info(const std::string& msg)    { write(LogLevel::INFO,    msg); }
    void log_warn(const std::string& msg)    { write(LogLevel::WARNING, msg); }
    void log_error(const std::string& msg)   { write(LogLevel::ERROR,   msg); }

    LogLevel min_level() const { return min_level_; }

private:
    // Format and emit a log line if the level is enabled.
    void write(LogLevel level, const std::string& msg) {
        if (static_cast<int>(level) < static_cast<int>(min_level_)) return;

        std::string line = format_line(level, msg);

        std::lock_guard<std::mutex> lock(mutex_);
        std::cout << line << "\n";
        file_stream_ << line << "\n";
        file_stream_.flush();
    }

    // Produce: [YYYY-MM-DD HH:MM:SS.mmm] [LEVEL] [tutorial] message
    std::string format_line(LogLevel level, const std::string& msg) const {
        auto now    = std::chrono::system_clock::now();
        auto ms     = std::chrono::duration_cast<std::chrono::milliseconds>(
                          now.time_since_epoch()) % 1000;
        std::time_t tt = std::chrono::system_clock::to_time_t(now);
        std::tm     tm_info{};
        localtime_r(&tt, &tm_info);

        std::ostringstream oss;
        oss << "[";
        oss << std::put_time(&tm_info, "%Y-%m-%d %H:%M:%S");
        oss << "." << std::setfill('0') << std::setw(3) << ms.count();
        oss << "]";
        oss << " [" << std::setw(7) << std::left << log_level_to_string(level) << "]";
        oss << " [" << tutorial_name_ << "]";
        oss << " " << msg;
        return oss.str();
    }

    LogLevel     min_level_;
    std::string  tutorial_name_;
    std::ofstream file_stream_;
    mutable std::mutex mutex_;
};

} // namespace cuda_tutorials

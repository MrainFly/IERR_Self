#include "runtime/logging.h"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>

#include "ierr/error.h"

namespace ierr {

namespace {
LogLevel g_level = [] {
    const char* env = std::getenv("IERR_LOG");
    if (!env) return LogLevel::Info;
    if (!std::strcmp(env, "debug")) return LogLevel::Debug;
    if (!std::strcmp(env, "info"))  return LogLevel::Info;
    if (!std::strcmp(env, "warn"))  return LogLevel::Warn;
    if (!std::strcmp(env, "error")) return LogLevel::Error;
    return LogLevel::Info;
}();
std::mutex g_mu;

const char* lvl_name(LogLevel l) {
    switch (l) {
        case LogLevel::Debug: return "DEBUG";
        case LogLevel::Info:  return "INFO";
        case LogLevel::Warn:  return "WARN";
        case LogLevel::Error: return "ERROR";
    }
    return "?";
}
} // namespace

LogLevel log_level() { return g_level; }
void set_log_level(LogLevel lvl) { g_level = lvl; }

void log_emit(LogLevel lvl, const std::string& msg) {
    std::lock_guard<std::mutex> lk(g_mu);
    std::ostream& os = (lvl >= LogLevel::Warn) ? std::cerr : std::cout;
    os << "[ierr][" << lvl_name(lvl) << "] " << msg << '\n';
}

} // namespace ierr

extern "C" const char* ierr_status_str(ierr_status_t s) {
    switch (s) {
        case IERR_OK:                   return "OK";
        case IERR_ERR_INVALID_ARG:      return "INVALID_ARG";
        case IERR_ERR_OUT_OF_MEMORY:    return "OUT_OF_MEMORY";
        case IERR_ERR_NOT_FOUND:        return "NOT_FOUND";
        case IERR_ERR_NOT_IMPLEMENTED:  return "NOT_IMPLEMENTED";
        case IERR_ERR_DEVICE:           return "DEVICE";
        case IERR_ERR_TIMEOUT:          return "TIMEOUT";
        case IERR_ERR_INTERNAL:         return "INTERNAL";
    }
    return "UNKNOWN";
}

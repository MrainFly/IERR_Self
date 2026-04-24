// Tiny logging helper – avoids any third-party dep.
#ifndef IERR_INTERNAL_LOGGING_H
#define IERR_INTERNAL_LOGGING_H

#include <sstream>
#include <string>

namespace ierr {

enum class LogLevel { Debug = 0, Info = 1, Warn = 2, Error = 3 };

void log_emit(LogLevel lvl, const std::string& msg);
LogLevel log_level();
void set_log_level(LogLevel lvl);

#define IERR_LOG(LVL, EXPR)                                       \
    do {                                                          \
        if (::ierr::log_level() <= (LVL)) {                       \
            std::ostringstream _oss;                              \
            _oss << EXPR;                                         \
            ::ierr::log_emit((LVL), _oss.str());                  \
        }                                                         \
    } while (0)

#define IERR_DEBUG(EXPR) IERR_LOG(::ierr::LogLevel::Debug, EXPR)
#define IERR_INFO(EXPR)  IERR_LOG(::ierr::LogLevel::Info,  EXPR)
#define IERR_WARN(EXPR)  IERR_LOG(::ierr::LogLevel::Warn,  EXPR)
#define IERR_ERROR(EXPR) IERR_LOG(::ierr::LogLevel::Error, EXPR)

} // namespace ierr

#endif

#ifndef CANDLE_VIDEO_H
#define CANDLE_VIDEO_H

#include <stdint.h>

typedef int32_t (*candle_video_generate_t)(const char* config_json);
typedef const char* (*candle_last_error_t)();
typedef const char* (*candle_binding_version_t)();

static inline int32_t call_candle_video_generate(void* f, const char* config_json) {
    return ((candle_video_generate_t)f)(config_json);
}

static inline const char* call_candle_last_error(void* f) {
    return ((candle_last_error_t)f)();
}

static inline const char* call_candle_binding_version(void* f) {
    return ((candle_binding_version_t)f)();
}

#endif

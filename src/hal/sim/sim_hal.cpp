// PC-side HAL backend: simulates a multi-core NPU using std::thread workers,
// std::condition_variable for events, and aligned heap allocations for
// "device" memory.
//
// Responsibilities (mirrors what the real driver+firmware would do):
//   * One worker thread per virtual core, each with its own command queue.
//   * A Stream is a completion-tracked task pool: every submission is
//     dispatched to a core immediately (round-robin or pinned), and the
//     stream maintains submitted/done counters. This preserves fan-out
//     parallelism across cores; the Host Runtime is responsible for ordering
//     between dependent ops by calling stream_synchronize / events.
//   * Events are post-condition fences: stream_record_event captures the
//     current submitted-count, and waiting on the event blocks until the
//     stream's done-count reaches that snapshot.
//   * Memcpy is just a std::memcpy issued by a worker, going through the
//     same submission path (same DMA-like ordering semantics).

#include "ierr/hal.h"

#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#if defined(_WIN32)
  #include <malloc.h>
#endif

namespace {

constexpr uint32_t kAlign = 64; // emulate NPU buffer alignment

void* aligned_alloc_compat(size_t bytes) {
    if (bytes == 0) bytes = 1;
    // round up to multiple of alignment (required by std::aligned_alloc).
    size_t rounded = (bytes + kAlign - 1) & ~(static_cast<size_t>(kAlign) - 1);
#if defined(_WIN32)
    return _aligned_malloc(rounded, kAlign);
#else
    return std::aligned_alloc(kAlign, rounded);
#endif
}
void aligned_free_compat(void* p) {
#if defined(_WIN32)
    _aligned_free(p);
#else
    std::free(p);
#endif
}

struct Buffer {
    void* ptr = nullptr;
    size_t bytes = 0;
};

// A command sitting in a worker queue. Carries a pointer back to its owning
// stream so the worker can bump done_count + notify on completion.
struct Stream;
struct Job {
    ierr_hal_kernel_fn fn;
    void* user;
    uint32_t core_id;
    Stream* stream;
};

class Worker {
public:
    Worker(uint32_t id) : id_(id), stop_(false) {
        thread_ = std::thread([this] { Loop(); });
    }
    ~Worker() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            stop_ = true;
        }
        cv_.notify_all();
        if (thread_.joinable()) thread_.join();
    }

    void Submit(Job j) {
        {
            std::lock_guard<std::mutex> lk(mu_);
            queue_.push(std::move(j));
        }
        cv_.notify_one();
    }

private:
    void Loop();

    uint32_t id_;
    std::thread thread_;
    std::mutex mu_;
    std::condition_variable cv_;
    std::queue<Job> queue_;
    bool stop_;
};

struct Device {
    std::vector<std::unique_ptr<Worker>> workers;
    uint32_t num_cores = 0;
    std::atomic<uint32_t> rr_counter{0}; // round-robin core picker
};

// Stream = task pool with completion tracking. The Host Runtime is responsible
// for enforcing inter-op dependencies via stream_synchronize / events.
struct Stream {
    Device* dev = nullptr;
    std::atomic<uint64_t> submitted{0};
    std::atomic<uint64_t> done{0};
    std::mutex mu;
    std::condition_variable cv;

    uint64_t Submit() { return submitted.fetch_add(1) + 1; }
    void NotifyDone() {
        done.fetch_add(1);
        std::lock_guard<std::mutex> lk(mu);
        cv.notify_all();
    }
    void WaitFor(uint64_t target) {
        std::unique_lock<std::mutex> lk(mu);
        cv.wait(lk, [&] { return done.load() >= target; });
    }
};

void Worker::Loop() {
    for (;;) {
        Job j;
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait(lk, [this] { return stop_ || !queue_.empty(); });
            if (stop_ && queue_.empty()) return;
            j = std::move(queue_.front());
            queue_.pop();
        }
        if (j.fn) j.fn(j.user, j.core_id);
        if (j.stream) j.stream->NotifyDone();
    }
}

struct Event {
    Stream* stream;
    uint64_t target;
};

uint32_t pick_num_cores() {
    if (const char* env = std::getenv("IERR_SIM_CORES")) {
        int v = std::atoi(env);
        if (v > 0 && v < 1024) return static_cast<uint32_t>(v);
    }
    unsigned hc = std::thread::hardware_concurrency();
    if (hc == 0) hc = 4;
    return hc;
}

// ---- HAL function implementations ------------------------------------------

ierr_status_t sim_device_open(ierr_hal_device_t* out_dev) {
    if (!out_dev) return IERR_ERR_INVALID_ARG;
    auto dev = new Device();
    dev->num_cores = pick_num_cores();
    dev->workers.reserve(dev->num_cores);
    for (uint32_t i = 0; i < dev->num_cores; ++i) {
        dev->workers.emplace_back(std::make_unique<Worker>(i));
    }
    *out_dev = reinterpret_cast<ierr_hal_device_t>(dev);
    return IERR_OK;
}

ierr_status_t sim_device_close(ierr_hal_device_t dev) {
    if (!dev) return IERR_ERR_INVALID_ARG;
    delete reinterpret_cast<Device*>(dev);
    return IERR_OK;
}

ierr_status_t sim_device_query_caps(ierr_hal_device_t dev, ierr_hal_caps_t* out) {
    if (!dev || !out) return IERR_ERR_INVALID_ARG;
    auto* d = reinterpret_cast<Device*>(dev);
    out->num_cores = d->num_cores;
    out->mem_align_bytes = kAlign;
    out->name = "sim";
    return IERR_OK;
}

ierr_status_t sim_mem_alloc(ierr_hal_device_t dev, size_t bytes, ierr_hal_buffer_t* out) {
    if (!dev || !out) return IERR_ERR_INVALID_ARG;
    auto* b = new Buffer();
    b->ptr = aligned_alloc_compat(bytes);
    if (!b->ptr) { delete b; return IERR_ERR_OUT_OF_MEMORY; }
    b->bytes = bytes;
    *out = reinterpret_cast<ierr_hal_buffer_t>(b);
    return IERR_OK;
}

ierr_status_t sim_mem_free(ierr_hal_buffer_t buf) {
    if (!buf) return IERR_OK;
    auto* b = reinterpret_cast<Buffer*>(buf);
    aligned_free_compat(b->ptr);
    delete b;
    return IERR_OK;
}

void* sim_mem_map(ierr_hal_buffer_t buf) {
    if (!buf) return nullptr;
    return reinterpret_cast<Buffer*>(buf)->ptr;
}

size_t sim_mem_size(ierr_hal_buffer_t buf) {
    if (!buf) return 0;
    return reinterpret_cast<Buffer*>(buf)->bytes;
}

ierr_status_t sim_stream_create(ierr_hal_device_t dev, ierr_hal_stream_t* out) {
    if (!dev || !out) return IERR_ERR_INVALID_ARG;
    auto* s = new Stream();
    s->dev = reinterpret_cast<Device*>(dev);
    *out = reinterpret_cast<ierr_hal_stream_t>(s);
    return IERR_OK;
}

ierr_status_t sim_stream_destroy(ierr_hal_stream_t s) {
    if (!s) return IERR_OK;
    auto* st = reinterpret_cast<Stream*>(s);
    st->WaitFor(st->submitted.load());
    delete st;
    return IERR_OK;
}

ierr_status_t sim_stream_submit(ierr_hal_stream_t s, const ierr_hal_command_t* cmd) {
    if (!s || !cmd) return IERR_ERR_INVALID_ARG;
    auto* st = reinterpret_cast<Stream*>(s);
    auto* dev = st->dev;

    st->Submit(); // bump submitted-count
    uint32_t core = (cmd->preferred_core >= 0)
        ? (static_cast<uint32_t>(cmd->preferred_core) % dev->num_cores)
        : (dev->rr_counter.fetch_add(1) % dev->num_cores);

    Job j;
    j.fn = cmd->fn;
    j.user = cmd->user;
    j.core_id = core;
    j.stream = st;
    dev->workers[core]->Submit(std::move(j));
    return IERR_OK;
}

ierr_status_t sim_stream_memcpy(ierr_hal_stream_t s, ierr_memcpy_dir_t /*dir*/,
                                ierr_hal_buffer_t dst, size_t dst_off,
                                ierr_hal_buffer_t src, size_t src_off,
                                size_t bytes) {
    if (!s) return IERR_ERR_INVALID_ARG;
    if (bytes == 0) return IERR_OK;

    // src and dst can be either Buffer* (device) or raw host pointers depending
    // on direction. To keep the API uniform we always take Buffer handles; the
    // runtime wraps host buffers when needed. For sim, all memory is on the
    // host so the direction is informational only.
    auto* dst_b = reinterpret_cast<Buffer*>(dst);
    auto* src_b = reinterpret_cast<Buffer*>(src);
    if (!dst_b || !src_b) return IERR_ERR_INVALID_ARG;
    if (dst_off + bytes > dst_b->bytes) return IERR_ERR_INVALID_ARG;
    if (src_off + bytes > src_b->bytes) return IERR_ERR_INVALID_ARG;

    struct Args { void* d; const void* s; size_t n; };
    auto* args = new Args{
        static_cast<char*>(dst_b->ptr) + dst_off,
        static_cast<const char*>(src_b->ptr) + src_off,
        bytes
    };
    ierr_hal_command_t cmd;
    cmd.fn = [](void* u, uint32_t /*core*/) {
        auto* a = static_cast<Args*>(u);
        std::memcpy(a->d, a->s, a->n);
        delete a;
    };
    cmd.user = args;
    cmd.preferred_core = -1;
    return sim_stream_submit(s, &cmd);
}

ierr_status_t sim_stream_record_event(ierr_hal_stream_t s, ierr_hal_event_t* out) {
    if (!s || !out) return IERR_ERR_INVALID_ARG;
    auto* st = reinterpret_cast<Stream*>(s);
    auto* e = new Event{st, st->submitted.load()};
    *out = reinterpret_cast<ierr_hal_event_t>(e);
    return IERR_OK;
}

ierr_status_t sim_stream_wait_event(ierr_hal_stream_t s, ierr_hal_event_t e) {
    if (!s || !e) return IERR_ERR_INVALID_ARG;
    auto* ev = reinterpret_cast<Event*>(e);
    // Cross-stream wait: simply block here. Correct for our usage because the
    // Host Runtime only inserts these as synchronization points between ops.
    ev->stream->WaitFor(ev->target);
    return IERR_OK;
}

ierr_status_t sim_stream_synchronize(ierr_hal_stream_t s) {
    if (!s) return IERR_ERR_INVALID_ARG;
    auto* st = reinterpret_cast<Stream*>(s);
    st->WaitFor(st->submitted.load());
    return IERR_OK;
}

ierr_status_t sim_event_destroy(ierr_hal_event_t e) {
    delete reinterpret_cast<Event*>(e);
    return IERR_OK;
}

ierr_status_t sim_event_synchronize(ierr_hal_event_t e) {
    if (!e) return IERR_ERR_INVALID_ARG;
    auto* ev = reinterpret_cast<Event*>(e);
    ev->stream->WaitFor(ev->target);
    return IERR_OK;
}

const ierr_hal_t kSimHal = {
    sim_device_open, sim_device_close, sim_device_query_caps,
    sim_mem_alloc, sim_mem_free, sim_mem_map, sim_mem_size,
    sim_stream_create, sim_stream_destroy, sim_stream_submit, sim_stream_memcpy,
    sim_stream_record_event, sim_stream_wait_event, sim_stream_synchronize,
    sim_event_destroy, sim_event_synchronize,
};

// NPU stub – the real driver isn't available in this skeleton.
ierr_status_t npu_not_impl(ierr_hal_device_t*) { return IERR_ERR_NOT_IMPLEMENTED; }
const ierr_hal_t kNpuStub = {
    npu_not_impl,
    nullptr, nullptr,
    nullptr, nullptr, nullptr, nullptr,
    nullptr, nullptr, nullptr, nullptr,
    nullptr, nullptr, nullptr,
    nullptr, nullptr,
};

} // namespace

extern "C" const ierr_hal_t* ierr_hal_get_sim(void) { return &kSimHal; }
extern "C" const ierr_hal_t* ierr_hal_get_npu(void) { return &kNpuStub; }

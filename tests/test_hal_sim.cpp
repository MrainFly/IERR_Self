// Tests for the simulator HAL: device caps, memory, streams, events.
#include <atomic>
#include <chrono>
#include <thread>

#include "ierr/hal.h"
#include "test_harness.h"

IERR_TEST(SimHal_DeviceOpenAndCaps) {
    const ierr_hal_t* h = ierr_hal_get_sim();
    IERR_EXPECT(h != nullptr);
    ierr_hal_device_t dev = nullptr;
    IERR_EXPECT_EQ(h->device_open(&dev), IERR_OK);
    ierr_hal_caps_t caps{};
    IERR_EXPECT_EQ(h->device_query_caps(dev, &caps), IERR_OK);
    IERR_EXPECT(caps.num_cores >= 1);
    IERR_EXPECT(caps.mem_align_bytes >= 16);
    IERR_EXPECT(caps.name != nullptr);
    h->device_close(dev);
}

IERR_TEST(SimHal_MemAllocFreeMap) {
    const ierr_hal_t* h = ierr_hal_get_sim();
    ierr_hal_device_t dev; h->device_open(&dev);
    ierr_hal_buffer_t buf = nullptr;
    IERR_EXPECT_EQ(h->mem_alloc(dev, 1024, &buf), IERR_OK);
    void* p = h->mem_map(buf);
    IERR_EXPECT(p != nullptr);
    IERR_EXPECT(h->mem_size(buf) == 1024);
    // Alignment check.
    IERR_EXPECT((reinterpret_cast<uintptr_t>(p) % 64) == 0);
    h->mem_free(buf);
    h->device_close(dev);
}

IERR_TEST(SimHal_StreamRunsCommandsAndSyncs) {
    const ierr_hal_t* h = ierr_hal_get_sim();
    ierr_hal_device_t dev; h->device_open(&dev);
    ierr_hal_stream_t s; h->stream_create(dev, &s);

    std::atomic<int> counter{0};
    constexpr int kN = 200;
    for (int i = 0; i < kN; ++i) {
        ierr_hal_command_t cmd;
        cmd.fn = [](void* u, uint32_t) {
            auto* c = static_cast<std::atomic<int>*>(u);
            c->fetch_add(1);
        };
        cmd.user = &counter;
        cmd.preferred_core = -1;
        IERR_EXPECT_EQ(h->stream_submit(s, &cmd), IERR_OK);
    }
    IERR_EXPECT_EQ(h->stream_synchronize(s), IERR_OK);
    IERR_EXPECT_EQ(counter.load(), kN);

    h->stream_destroy(s);
    h->device_close(dev);
}

IERR_TEST(SimHal_EventReachesTarget) {
    const ierr_hal_t* h = ierr_hal_get_sim();
    ierr_hal_device_t dev; h->device_open(&dev);
    ierr_hal_stream_t s; h->stream_create(dev, &s);

    std::atomic<int> counter{0};
    for (int i = 0; i < 50; ++i) {
        ierr_hal_command_t cmd;
        cmd.fn = [](void* u, uint32_t) {
            std::this_thread::sleep_for(std::chrono::microseconds(50));
            static_cast<std::atomic<int>*>(u)->fetch_add(1);
        };
        cmd.user = &counter;
        cmd.preferred_core = -1;
        h->stream_submit(s, &cmd);
    }
    ierr_hal_event_t ev;
    IERR_EXPECT_EQ(h->stream_record_event(s, &ev), IERR_OK);
    IERR_EXPECT_EQ(h->event_synchronize(ev), IERR_OK);
    IERR_EXPECT_EQ(counter.load(), 50);
    h->event_destroy(ev);

    h->stream_destroy(s);
    h->device_close(dev);
}

IERR_TEST(SimHal_NpuStubReportsNotImplemented) {
    const ierr_hal_t* h = ierr_hal_get_npu();
    IERR_EXPECT(h != nullptr);
    ierr_hal_device_t dev = nullptr;
    IERR_EXPECT_EQ(h->device_open(&dev), IERR_ERR_NOT_IMPLEMENTED);
}

#include "llaisys/models/qwen2.h"
#include "../../models/qwen2/model.hpp"
#include <stdexcept>
#include <string>

struct LlaisysQwen2Model {
    llaisys::models::qwen2::Model model;
};
namespace {
thread_local std::string last_error;
template <typename Result, typename Function>
Result guarded(Result failure, Function function) {
    last_error.clear();
    try {
        return function();
    } catch (const std::exception &error) {
        last_error = error.what();
    } catch (...) { last_error = "Unknown Qwen2 backend error"; }
    return failure;
}
void requireModel(LlaisysQwen2Model *model) {
    if (!model) {
        throw std::invalid_argument("Qwen2 model is null or closed");
    }
}
} // namespace
extern "C" {
LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta,
                                           llaisysDeviceType_t device,
                                           const int *device_ids, int ndevice) {
    return guarded<LlaisysQwen2Model *>(nullptr, [&] {
        if (!meta) {
            throw std::invalid_argument("Qwen2 meta is null");
        }
        return new LlaisysQwen2Model{
            llaisys::models::qwen2::Model(*meta, device, device_ids, ndevice)};
    });
}
void llaisysQwen2ModelDestroy(LlaisysQwen2Model *model) {
    guarded<int>(-1, [&] {
        delete model;
        return 0;
    });
}
LlaisysQwen2Weights *llaisysQwen2ModelWeights(LlaisysQwen2Model *model) {
    return guarded<LlaisysQwen2Weights *>(nullptr, [&] {
        requireModel(model);
        return model->model.weights();
    });
}
void llaisysQwen2ModelReset(LlaisysQwen2Model *model) {
    guarded<int>(-1, [&] {
        requireModel(model);
        model->model.reset();
        return 0;
    });
}
int64_t llaisysQwen2ModelInfer(LlaisysQwen2Model *model,
                               const int64_t *token_ids, size_t ntoken) {
    return guarded<int64_t>(-1, [&] {
        requireModel(model);
        return model->model.infer(token_ids, ntoken);
    });
}
int llaisysQwen2ModelState(LlaisysQwen2Model *model, LlaisysQwen2State *state) {
    return guarded<int>(-1, [&] {
        requireModel(model);
        if (!state) {
            throw std::invalid_argument("Qwen2 state pointer is null");
        }
        *state = model->model.state();
        return 0;
    });
}
const char *llaisysQwen2LastError() { return last_error.c_str(); }
}

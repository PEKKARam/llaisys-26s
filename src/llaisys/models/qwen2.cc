#include "llaisys/models/qwen2.h"

#include "../../models/qwen2/model.hpp"

struct LlaisysQwen2Model {
    llaisys::models::qwen2::Model model;
};

__C {

    struct LlaisysQwen2Model *llaisysQwen2ModelCreate(
        const LlaisysQwen2Meta *meta,
        llaisysDeviceType_t device,
        const int *device_ids,
        int ndevice) {
        if (meta == nullptr) {
            return nullptr;
        }
        return new LlaisysQwen2Model{llaisys::models::qwen2::Model(*meta, device, device_ids, ndevice)};
    }

    void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model *model) {
        delete model;
    }

    struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model *model) {
        return model == nullptr ? nullptr : model->model.weights();
    }

    void llaisysQwen2ModelReset(struct LlaisysQwen2Model *model) {
        if (model != nullptr) {
            model->model.reset();
        }
    }

    int64_t llaisysQwen2ModelInfer(
        struct LlaisysQwen2Model *model,
        const int64_t *token_ids,
        size_t ntoken) {
        if (model == nullptr) {
            return -1;
        }
        return model->model.infer(token_ids, ntoken);
    }
}

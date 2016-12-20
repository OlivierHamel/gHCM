
#include "field_loaders.h"
#include "util_files.h"
#include "util_opencl.h"


namespace {
struct StbImage {
    int     w = -1, h = -1;
    float*  data/*[w*h]*/ = nullptr;

    StbImage()                  = default;
    StbImage(StbImage const&)   = delete;
    StbImage(StbImage&& m   ) { *this = std::move(m); }
    ~StbImage()               { stbi_image_free(data); }

    StbImage& operator=(StbImage&& m) {
        std::swap(h     , m.h   );
        std::swap(w     , m.w   );
        std::swap(data  , m.data);
        return *this;
    }

    static optional<StbImage> file_load(char const* const file_path
                                       ,int         const channels = STBI_rgb) {
        assert(file_path && "Expected a file path for img.");
        if (!file_path) return {};

        auto const file_data = file_read_all<stbi_uc>(file_path);
        if (!file_data) {
            std::cerr << "Failed to read image file: " << file_path << "\n";
            return {};
        }

        StbImage img {};
        img.data = stbi_loadf_from_memory(file_data->data(), file_data->size()
                                         ,&img.w, &img.h, nullptr, channels);
        if (!img.data) {
            std::cerr << "Failed to load image grayscale: " << file_path << "\n";
            std::cerr << "\tError: " << stbi_failure_reason() << "\n";
            return {};
        }

        assert(0 < img.w);
        assert(0 < img.h);
        return std::move(img);
    }
};

}


optional<Field2D<float>> field2d_img_monochannel(char const* const file_path) {
    auto const img = StbImage::file_load(file_path, STBI_grey);
    if (!img) return {};

    return Field2D<float>({ img->w, img->h }, img->data);
}

// \todo variant which just leaves the buffer on the GPU?
optional<Field2D<float>> field2d_img_rgb_luminosity(char const* const file_path) {
    auto const img = StbImage::file_load(file_path, STBI_rgb);
    if (!img) return {};

    auto const all_gpus     = cl_platform_devices(cl::Platform::getDefault(), CL_DEVICE_TYPE_GPU);
    auto const device_info  = cl_device_select_first(all_gpus, CL_DEVICE_TYPE_GPU);
    if (!device_info) {
        std::cerr << "No GPUs available for transcoding to grayscale?\n";
        exit(-1); // Don't bother handling this, we're here for solving Eikonals w/ a GPU.
    }

    cl_uint2 img_size = { img->w, img->h };
    auto const ctx          = cl::Context(device_info->device);
    auto       cmd_queue    = cl::CommandQueue(ctx, device_info->device, CL_QUEUE_PROFILING_ENABLE);
    auto const program      = cl_util_build(ctx, { device_info->device }
                                           ," -x clc++ -cl-mad-enable -I \"./data\""
                                           ,"data/img_proc_rgb_luminosity");
    if (!program) {
        std::cerr << "Failed to compile rgb luminosity kernel.\n";
        exit(-1); // don't handle, this isn't meant to be recoverable
    }

    auto       field        = Field2D<float>({ img->w, img->h });
    auto const num_cells    = img->w * img->h;
    auto const buf_input    = cl::Buffer(ctx, CL_MEM_READ_ONLY  | CL_MEM_USE_HOST_PTR, num_cells * 3 * sizeof(float), img->data);
    auto const buf_output   = cl::Buffer(ctx, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR
                                        ,num_cells * sizeof(float)
                                        ,field.backing_store().data());
    auto       kernel       = cl::KernelFunctor<cl::Buffer /*input*/
                                               ,cl::Buffer /*output*/
                                               ,cl_uint2   /*image_size*/
                                               >(*program, "img_proc_rgb_luminosity");
    auto const kernel_done  = kernel(cl::EnqueueArgs(cmd_queue, cl::NDRange(num_cells))
                                    ,buf_input, buf_output, img_size);
    kernel_done.wait();
    
    
    auto const events_await = cl::vector<cl::Event> { kernel_done };
    cmd_queue.enqueueReadBuffer(buf_output, CL_TRUE, 0
                               ,sizeof(float) * num_cells, field.backing_store().data()
                               ,&events_await);
    return field;
}
// Copyright © 2023-2024 Apple Inc.

#include <mach/mach_time.h>
#include <atomic>
#include <cstdlib>
#include <mutex>
#include <sstream>

#include <fmt/format.h>

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include "mlx/backend/common/utils.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/event.h"
#include "mlx/backend/metal/metal.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/utils.h"

namespace std {

// Required for putting the pointer in unordered_set.
template <class T>
struct hash<NS::SharedPtr<T>> {
  size_t operator()(const NS::SharedPtr<T>& p) const {
    return std::hash<T*>{}(p.get());
  }
};

} // namespace std

namespace mlx::core::metal {

namespace {

constexpr const char* default_mtllib_path = METAL_PATH;

void set_compile_options(
    MTL::CompileOptions* mtl_options,
    const CompileOptions& compile_options) {
  if (__builtin_available(macOS 15, iOS 18, tvOS 18, visionOS 2, *)) {
    switch (compile_options.math_mode) {
      case MathMode::Safe:
        mtl_options->setMathMode(MTL::MathModeSafe);
        break;
      case MathMode::Relaxed:
        mtl_options->setMathMode(MTL::MathModeRelaxed);
        break;
      case MathMode::Fast:
        mtl_options->setMathMode(MTL::MathModeFast);
        break;
      default:
        throw std::invalid_argument("[metal::Device] Invalid math mode.");
    }
  } else {
    if (compile_options.math_mode == MathMode::Relaxed) {
      throw std::runtime_error(
          "[metal::Device] Metal math mode `relaxed` requires macOS 15, "
          "iOS 18, tvOS 18, or visionOS 2.");
    }
    mtl_options->setFastMathEnabled(
        compile_options.math_mode == MathMode::Fast);
  }
}

auto get_metal_version() {
  auto get_metal_version_ = []() {
    if (__builtin_available(macOS 27, iOS 27, tvOS 27, visionOS 27, *)) {
      // TODO: Use MTL::LanguageVersion4_1 after metal-cpp_27 is released.
      return static_cast<MTL::LanguageVersion>((4 << 16) + 1);
    } else if (__builtin_available(macOS 26, iOS 26, tvOS 26, visionOS 26, *)) {
      return MTL::LanguageVersion4_0;
    } else if (__builtin_available(macOS 15, iOS 18, tvOS 18, visionOS 2, *)) {
      return MTL::LanguageVersion3_2;
    } else {
      return MTL::LanguageVersion3_1;
    }
  };
  static auto metal_version_ = get_metal_version_();
  return metal_version_;
}

NS::SharedPtr<MTL::Device> load_device() {
  auto pool = new_scoped_memory_pool();
  auto devices = NS::TransferPtr(MTL::CopyAllDevices());
  // In headless, sandboxed, or virtualized macOS sessions CopyAllDevices()
  // returns an empty NSArray. Indexing object(0) on an empty array raises
  // an unrecoverable NSRangeException, so guard the access and fall back to
  // CreateSystemDefaultDevice (which can also return null).
  MTL::Device* first = nullptr;
  if (devices && devices->count() > 0) {
    first = static_cast<MTL::Device*>(devices->object(0));
  }
  auto device = first ? NS::RetainPtr(first)
                      : NS::TransferPtr(MTL::CreateSystemDefaultDevice());
  if (!device) {
    throw std::runtime_error(
        "[metal::load_device] No Metal device available. This typically "
        "occurs in headless, sandboxed, or virtualized macOS sessions "
        "where the GPU is not accessible.");
  }
  return device;
}

std::pair<MTL::Library*, NS::Error*> load_library_from_path(
    MTL::Device* device,
    const char* path) {
  auto library = NS::String::string(path, NS::UTF8StringEncoding);
  NS::Error* error;
  auto lib = device->newLibrary(library, &error);

  return std::make_pair(lib, error);
}

#ifdef SWIFTPM_BUNDLE
MTL::Library* try_load_bundle(
    MTL::Device* device,
    NS::URL* url,
    const std::string& lib_name) {
  std::string bundle_path = std::string(url->fileSystemRepresentation()) + "/" +
      SWIFTPM_BUNDLE + ".bundle";
  auto bundle = NS::Bundle::alloc()->init(
      NS::String::string(bundle_path.c_str(), NS::UTF8StringEncoding));
  if (bundle != nullptr) {
    std::string resource_path =
        std::string(bundle->resourceURL()->fileSystemRepresentation()) + "/" +
        lib_name + ".metallib";
    auto [lib, error] = load_library_from_path(device, resource_path.c_str());
    if (lib) {
      return lib;
    }
  }
  return nullptr;
}

MTL::Library* try_load_framework(
    MTL::Device* device,
    NS::URL* url,
    const std::string& lib_name) {
  std::string resource_path = std::string(url->fileSystemRepresentation()) +
      "/" + lib_name + ".metallib";
  auto [lib, error] = load_library_from_path(device, resource_path.c_str());
  if (lib) {
    return lib;
  }
  return nullptr;
}
#endif

// Firstly, search for the metallib in the same path as this binary
std::pair<MTL::Library*, NS::Error*> load_colocated_library(
    MTL::Device* device,
    const std::string& relative_path) {
  auto path = current_binary_dir() / relative_path;
  if (!path.has_extension()) {
    path.replace_extension(".metallib");
  }

  return load_library_from_path(device, path.c_str());
}

std::pair<MTL::Library*, NS::Error*> load_swiftpm_library(
    MTL::Device* device,
    const std::string& lib_name) {
#ifdef SWIFTPM_BUNDLE
  MTL::Library* library =
      try_load_bundle(device, NS::Bundle::mainBundle()->bundleURL(), lib_name);
  if (library != nullptr) {
    return {library, nullptr};
  }
  auto bundles = NS::Bundle::allBundles();
  for (int i = 0, c = (int)bundles->count(); i < c; i++) {
    auto bundle = reinterpret_cast<NS::Bundle*>(bundles->object(i));
    library = try_load_bundle(device, bundle->resourceURL(), lib_name);
    if (library != nullptr) {
      return {library, nullptr};
    }
  }
  // if SWIFTPM_BUNDLE is a framework identifier, try loading from that
  auto frameworks = NS::Bundle::allFrameworks();
  for (int i = 0, c = (int)frameworks->count(); i < c; i++) {
    const auto bundle = reinterpret_cast<NS::Bundle*>(frameworks->object(i));
    const auto identifier = bundle->bundleIdentifier();
    if (identifier != nullptr &&
        !strcmp(identifier->utf8String(), SWIFTPM_BUNDLE)) {
      library = try_load_framework(device, bundle->resourceURL(), lib_name);
      if (library != nullptr) {
        return {library, nullptr};
      }
    }
  }
#endif
  return {nullptr, nullptr};
}

MTL::Library* load_default_library(MTL::Device* device) {
  // Check override path before automatic lookup
  if (!get_metallib_path().empty()) {
    auto [lib, error] =
        load_library_from_path(device, get_metallib_path().c_str());
    if (!lib) {
      throw std::runtime_error(
          fmt::format(
              "Can not load metallib from specified location \"{}\": {}.",
              get_metallib_path(),
              error->localizedDescription()->utf8String()));
    }
    return lib;
  }

  NS::Error* error[5];
  MTL::Library* lib;
  // First try the colocated mlx.metallib
  std::tie(lib, error[0]) = load_colocated_library(device, "mlx");
  if (lib) {
    return lib;
  }

  std::tie(lib, error[1]) = load_colocated_library(device, "Resources/mlx");
  if (lib) {
    return lib;
  }

  // Then try default.metallib in a SwiftPM bundle if we have one
  std::tie(lib, error[2]) = load_swiftpm_library(device, "default");
  if (lib) {
    return lib;
  }

  // Try lo load resources from Framework resources if SwiftPM wrapped as a
  // dynamic framework.
  std::tie(lib, error[3]) = load_colocated_library(device, "Resources/default");
  if (lib) {
    return lib;
  }

  // Finally try default_mtllib_path
  std::tie(lib, error[4]) = load_library_from_path(device, default_mtllib_path);
  if (!lib) {
    std::ostringstream msg;
    msg << "Failed to load the default metallib. ";
    for (int i = 0; i < 5; i++) {
      if (error[i] != nullptr) {
        msg << error[i]->localizedDescription()->utf8String() << " ";
      }
    }
    throw std::runtime_error(msg.str());
  }
  return lib;
}

MTL::Library* load_library(
    MTL::Device* device,
    const std::string& lib_name,
    const std::string& lib_path) {
  // We have been given a path that ends in metallib so try to load it
  if (lib_path.size() > 9 &&
      std::equal(lib_path.end() - 9, lib_path.end(), ".metallib")) {
    auto [lib, error] = load_library_from_path(device, lib_path.c_str());
    if (!lib) {
      std::ostringstream msg;
      msg << "Failed to load the metallib from <" << lib_path << "> with error "
          << error->localizedDescription()->utf8String();
      throw std::runtime_error(msg.str());
    }
    return lib;
  }

  // We have been given a path so try to load from lib_path / lib_name.metallib
  if (lib_path.size() > 0) {
    std::string full_path = lib_path + "/" + lib_name + ".metallib";
    auto [lib, error] = load_library_from_path(device, full_path.c_str());
    if (!lib) {
      std::ostringstream msg;
      msg << "Failed to load the metallib from <" << full_path
          << "> with error " << error->localizedDescription()->utf8String();
      throw std::runtime_error(msg.str());
    }
    return lib;
  }

  // Try to load the colocated library
  {
    auto [lib, error] = load_colocated_library(device, lib_name);
    if (lib) {
      return lib;
    }
  }

  // Try to load the library from swiftpm
  {
    auto [lib, error] = load_swiftpm_library(device, lib_name);
    if (lib) {
      return lib;
    }
  }

  std::ostringstream msg;
  msg << "Failed to load the metallib " << lib_name << ".metallib. "
      << "We attempted to load it from <" << current_binary_dir() << "/"
      << lib_name << ".metallib>";
#ifdef SWIFTPM_BUNDLE
  msg << " and from the Swift PM bundle.";
#endif
  throw std::runtime_error(msg.str());
}

} // namespace

CommandEncoder::CommandEncoder(
    Device& d,
    int index,
    ResidencySets& residency_sets)
    : device_(d), residency_sets_(residency_sets) {
  auto pool = new_scoped_memory_pool();
  queue_ = NS::TransferPtr(device_.mtl_device()->newCommandQueue());
  if (!queue_) {
    throw std::runtime_error(
        "[metal::CommandEncoder] Failed to make new command queue.");
  }
  // Sets created later are attached in commit().
  residency_sets_.attach_new_sets(queue_.get(), sets_attached_);
  debug_set_stream_queue_label(queue_.get(), index);
  buffer_ = NS::RetainPtr(queue_->commandBufferWithUnretainedReferences());
}

CommandEncoder::~CommandEncoder() {
  exiting_ = true;
  synchronize();
  auto pool = new_scoped_memory_pool();
  buffer_.reset();
  queue_.reset();
}

void CommandEncoder::set_buffer(
    const MTL::Buffer* buf,
    int idx,
    int64_t offset /* = 0 */) {
  // Record as both input and output to ensure synchronization between command
  // buffers
  all_inputs_.insert((void*)buf);
  all_outputs_.insert((void*)buf);
  get_command_encoder()->setBuffer(buf, offset, idx);
}

void CommandEncoder::set_input_array(
    const array& a,
    int idx,
    int64_t offset /* = 0 */) {
  if (all_inputs_.insert(a.buffer().ptr()).second) {
    buffer_sizes_ += a.data_size();
  }
  auto r_buf = static_cast<MTL::Resource*>(const_cast<void*>(a.buffer().ptr()));
  next_inputs_.insert(r_buf);
  needs_barrier_ =
      needs_barrier_ | (prev_outputs_.find(r_buf) != prev_outputs_.end());
  auto a_buf = static_cast<const MTL::Buffer*>(a.buffer().ptr());
  get_command_encoder()->setBuffer(a_buf, a.offset() + offset, idx);
}

void CommandEncoder::set_output_array(
    array& a,
    int idx,
    int64_t offset /* = 0 */) {
  // Add barriers before adding the output to the output set
  set_input_array(a, idx, offset);
  register_output_array(a);
}

void CommandEncoder::register_output_array(const array& a) {
  all_outputs_.insert(a.buffer().ptr());

  auto buf = static_cast<MTL::Resource*>(const_cast<void*>(a.buffer().ptr()));
  if (concurrent_) {
    concurrent_outputs_.insert(buf);
  } else {
    next_outputs_.insert(buf);
    needs_barrier_ =
        needs_barrier_ | (prev_inputs_.find(buf) != prev_inputs_.end());
  }
}

void CommandEncoder::add_temporary(array arr) {
  temporaries_.push_back(std::move(arr));
}

void CommandEncoder::add_temporaries(std::vector<array> arrays) {
  temporaries_.insert(
      temporaries_.end(),
      std::make_move_iterator(arrays.begin()),
      std::make_move_iterator(arrays.end()));
}

void CommandEncoder::maybeInsertBarrier() {
  if (needs_barrier_) {
    get_command_encoder()->memoryBarrier(MTL::BarrierScopeBuffers);
    needs_barrier_ = false;
    // Preserve the hash tables' buckets for reuse across barrier epochs.
    prev_inputs_.swap(next_inputs_);
    prev_outputs_.swap(next_outputs_);
  } else {
    prev_inputs_.insert(next_inputs_.begin(), next_inputs_.end());
    prev_outputs_.insert(next_outputs_.begin(), next_outputs_.end());
  }
  next_inputs_.clear();
  next_outputs_.clear();
}

std::atomic<long> phone_dispatches{0};  // phone-store instrumentation: kernels encoded
extern "C" long mlx_phone_dispatches() { return phone_dispatches.load(); }

// phone-store GPU profiler (not upstream). Off unless mlx_phone_prof_set(1); no effect on what runs. mlx_phone_prof_set(2):
// the same, and every commit waits for its buffer (serial: with MLX_MAX_OPS_PER_BUFFER=0, one kernel at a time). When on, every
// dispatch records (kernel, tag) in its command buffer, and every committed buffer's completion records the buffer's
// commit time (host), GPUStartTime / GPUEndTime and kernelStartTime / kernelEndTime (all on the host clock,
// CACurrentMediaTime's mach time base) with its dispatches. mlx_phone_prof_dump writes it all as text:
//   K <id> <kernel name>              (every pipeline MLX created, named as MLX's kernel cache names it)
//   B <seq> <commit> <gpu_start> <gpu_end> <kernel_start> <kernel_end> <encoders> <id:tag,id:tag,...>
//   M <kind> <host time> <value>      (marks from mlx_phone_prof_mark)
namespace {
std::atomic<int> phone_prof_on{0};
std::atomic<int> phone_prof_tag{0};
std::atomic<long> phone_prof_seq{0};
std::mutex phone_prof_mtx;
std::unordered_map<const void*, std::string> phone_prof_names;
struct PhoneProfBuf {
  long seq;
  double commit, gs, ge, ks, ke;
  int encoders;
  std::vector<std::pair<const void*, int>> disp;
};
std::vector<PhoneProfBuf> phone_prof_bufs;
struct PhoneProfMark {
  int kind;
  double t;
  long v;
};
std::vector<PhoneProfMark> phone_prof_marks;
double phone_prof_now() {
  static mach_timebase_info_data_t tb{0, 0};
  if (tb.denom == 0) mach_timebase_info(&tb);
  return double(mach_absolute_time()) * double(tb.numer) / double(tb.denom) / 1e9;
}
} // namespace
void phone_prof_register(const void* pso, const std::string& name) {
  std::lock_guard<std::mutex> lk(phone_prof_mtx);
  phone_prof_names[pso] = name;
}
extern "C" void mlx_phone_prof_set(int on) { phone_prof_on.store(on); }
extern "C" void mlx_phone_prof_tag(int tag) { phone_prof_tag.store(tag); }
extern "C" double mlx_phone_prof_now() { return phone_prof_now(); }
extern "C" void mlx_phone_prof_mark(int kind, long v) {
  double t = phone_prof_now();
  std::lock_guard<std::mutex> lk(phone_prof_mtx);
  phone_prof_marks.push_back({kind, t, v});
}
extern "C" void mlx_phone_prof_clear() {
  std::lock_guard<std::mutex> lk(phone_prof_mtx);
  phone_prof_bufs.clear();
  phone_prof_marks.clear();
}
extern "C" long mlx_phone_prof_dump(const char* path) {
  std::lock_guard<std::mutex> lk(phone_prof_mtx);
  FILE* f = fopen(path, "w");
  if (!f) return -1;
  std::unordered_map<const void*, int> ids;
  for (auto& [p, n] : phone_prof_names) {
    int id = int(ids.size());
    ids[p] = id;
    fprintf(f, "K %d %s\n", id, n.c_str());
  }
  for (auto& b : phone_prof_bufs) {
    fprintf(f, "B %ld %.9f %.9f %.9f %.9f %.9f %d ", b.seq, b.commit, b.gs, b.ge, b.ks, b.ke, b.encoders);
    for (size_t i = 0; i < b.disp.size(); ++i) {
      auto it = ids.find(b.disp[i].first);
      fprintf(f, "%s%d:%d", i ? "," : "", it == ids.end() ? -1 : it->second, b.disp[i].second);
    }
    fprintf(f, "\n");
  }
  for (auto& m : phone_prof_marks) fprintf(f, "M %d %.9f %ld\n", m.kind, m.t, m.v);
  fclose(f);
  return long(phone_prof_bufs.size());
}

void CommandEncoder::dispatch_threadgroups(
    MTL::Size grid_dims,
    MTL::Size group_dims) {
  maybeInsertBarrier();
  buffer_ops_++;
  phone_dispatches++;
  if (phone_prof_on.load(std::memory_order_relaxed)) phone_disp_.emplace_back(phone_pso_, phone_prof_tag.load(std::memory_order_relaxed));
  get_command_encoder()->dispatchThreadgroups(grid_dims, group_dims);
}

void CommandEncoder::dispatch_threads(
    MTL::Size grid_dims,
    MTL::Size group_dims) {
  maybeInsertBarrier();
  buffer_ops_++;
  phone_dispatches++;
  if (phone_prof_on.load(std::memory_order_relaxed)) phone_disp_.emplace_back(phone_pso_, phone_prof_tag.load(std::memory_order_relaxed));
  get_command_encoder()->dispatchThreads(grid_dims, group_dims);
}

void CommandEncoder::barrier() {
  get_command_encoder()->memoryBarrier(MTL::BarrierScopeBuffers);
}

void CommandEncoder::end_encoding() {
  // Each command encoder has a unique fence. We also store a map of
  // all previous outputs of command encoders to their corresponding fence.
  // - The command encoder records its inputs and outputs.
  // - Wait on a fence if any inputs in the encoder are outputs of a previous
  //   encoder.
  // - Update the map of outputs to include this command encoder's outputs.
  // - Always signal this command encoders fence.
  // - Add a completion handler for this command encoder that removes outputs
  //   from the map to limit the growth of the map and avoid unnecessary waits
  // - Temporaries are a special case as they do not cross command encoder
  //   boundaries. These can be removed early from the encoders inputs and
  //   outputs since they don't need synchronization.
  if (!encoder_) {
    return;
  }

  // Remove temporaries from inputs and outputs.
  for (auto& t : temporaries_) {
    all_outputs_.erase(t.buffer().ptr());
    all_inputs_.erase(t.buffer().ptr());
  }

  // Keep references to the fences we waited on and put them in the completion
  // handler so they are not prematurely released.
  std::unordered_set<NS::SharedPtr<MTL::Fence>> waiting_on;
  {
    std::lock_guard lk(outputs_mtx_);
    for (auto& in : all_inputs_) {
      if (auto it = prev_ce_outputs_.find(in); it != prev_ce_outputs_.end()) {
        // If we've already waited on a fence, don't wait on it again.
        if (waiting_on.find(it->second) == waiting_on.end()) {
          encoder_->waitForFence(it->second.get());
          waiting_on.insert(it->second);
        }
      }
    }
    for (auto& out : all_outputs_) {
      prev_ce_outputs_[out] = fence_;
    }
  }

  encoder_->updateFence(fence_.get());
  buffer_->addCompletedHandler([this,
                                fence = std::move(fence_),
                                temporaries = std::move(temporaries_),
                                all_outputs = std::move(all_outputs_),
                                waiting_on = std::move(waiting_on)](
                                   MTL::CommandBuffer*) mutable {
    std::lock_guard lk(outputs_mtx_);
    for (auto& o : all_outputs) {
      if (auto it = prev_ce_outputs_.find(o); it != prev_ce_outputs_.end()) {
        if (it->second == fence) {
          prev_ce_outputs_.erase(it);
        }
      }
    }
  });

  encoder_->endEncoding();
  encoder_.reset();
  phone_encoders_++;
  needs_barrier_ = false;
  concurrent_ = false;
  prev_inputs_.clear();
  next_inputs_.clear();
  prev_outputs_.clear();
  next_outputs_.clear();
  concurrent_outputs_.clear();
  all_inputs_.clear();
}

void CommandEncoder::signal_event(Event event, uint64_t value) {
  end_encoding();
  buffer_->encodeSignalEvent(event.cast<EventImpl>().mtl_event(), value);
  signal_events_.push_back({std::move(event), value});
}

void CommandEncoder::wait_event(Event event, uint64_t value) {
  end_encoding();
  buffer_->encodeWait(event.cast<EventImpl>().mtl_event(), value);
  wait_events_.push_back(std::move(event));
}

bool CommandEncoder::needs_commit() const {
  auto [max_ops, max_mb] = device_.get_max_ops_mb_per_buffer();
  return (buffer_ops_ > max_ops) || ((buffer_sizes_ >> 20) > max_mb);
}

// phone-store instrumentation (not in upstream mlx): GPU busy time of every committed command buffer, as the union
// of [GPUStartTime, GPUEndTime] intervals, read from Swift through mlx_phone_gpu_stats(). No effect on what runs.
namespace {
std::mutex phone_gpu_mtx;
double phone_gpu_busy = 0.0, phone_gpu_last_end = 0.0;
long phone_gpu_buffers = 0;
} // namespace
extern "C" void mlx_phone_gpu_stats(double* busy_seconds, long* buffers) {
  std::lock_guard<std::mutex> lk(phone_gpu_mtx);
  *busy_seconds = phone_gpu_busy;
  *buffers = phone_gpu_buffers;
}

void CommandEncoder::commit(std::function<void()> completion) {
  // Metal locks a command buffer's residency at commit time, so attach any
  // sets created since the last commit first.
  residency_sets_.attach_new_sets(queue_.get(), sets_attached_);
  if (phone_prof_on.load(std::memory_order_relaxed)) {
    buffer_->addCompletedHandler(
        [disp = std::move(phone_disp_), encs = phone_encoders_, seq = phone_prof_seq++, tc = phone_prof_now()](
            MTL::CommandBuffer* cbuf) mutable {
          PhoneProfBuf b{seq, tc, cbuf->GPUStartTime(), cbuf->GPUEndTime(), cbuf->kernelStartTime(), cbuf->kernelEndTime(),
                         encs, std::move(disp)};
          std::lock_guard<std::mutex> lk(phone_prof_mtx);
          phone_prof_bufs.push_back(std::move(b));
        });
  }
  phone_disp_.clear();
  phone_encoders_ = 0;
  buffer_->addCompletedHandler(
      [&error_ = error_,
       wait_events = std::move(wait_events_),
       signal_events = std::move(signal_events_),
       completion = std::move(completion)](MTL::CommandBuffer* cbuf) mutable {
        {
          double t0 = cbuf->GPUStartTime(), t1 = cbuf->GPUEndTime();
          std::lock_guard<std::mutex> lk(phone_gpu_mtx);
          if (t1 > t0) {
            double s0 = t0 > phone_gpu_last_end ? t0 : phone_gpu_last_end;
            if (t1 > s0) phone_gpu_busy += t1 - s0;
            if (t1 > phone_gpu_last_end) phone_gpu_last_end = t1;
          }
          phone_gpu_buffers += 1;
        }
        if (completion) {
          completion();
        }
        // If any of the waited event has error in it, poison the encoder.
        for (auto& event : wait_events) {
          if (error_.store_if_valid(event.load_error())) {
            break;
          }
        }
        // Set error only when no error happended before, to preserve the
        // earliest error.
        bool has_error = error_.valid();
        if (!has_error && cbuf->status() == MTL::CommandBufferStatusError) {
          error_.set_message(
              std::make_shared<std::string>(fmt::format(
                  "[METAL] Command buffer execution failed: {}.",
                  cbuf->error()->localizedDescription()->utf8String())));
          has_error = true;
        }
        // Poison all the signaled events when error happened.
        if (has_error) {
          for (auto& [event, value] : signal_events) {
            event.set_error(error_);
          }
        }
        // Metal won't signal the events for us on error, manually signal them
        // to avoid infinite waiting.
        if (cbuf->status() == MTL::CommandBufferStatusError) {
          for (auto& [event, value] : signal_events) {
            event.cast<EventImpl>().signal(value);
          }
        }
      });
  buffer_->commit();
  if (phone_prof_on.load(std::memory_order_relaxed) == 2) {
    buffer_->waitUntilCompleted();  // phone-store GPU profiler, serial mode: one command buffer on the GPU at a time
  }
  buffer_ = NS::RetainPtr(queue_->commandBufferWithUnretainedReferences());
  buffer_ops_ = 0;
  buffer_sizes_ = 0;
}

void CommandEncoder::synchronize() {
  auto pool = new_scoped_memory_pool();
  auto cbuf = buffer_; // retained
  end_encoding();
  commit();
  cbuf->waitUntilCompleted();

  if (!exiting_) {
    error_.check();
  }
}

MTL::ComputeCommandEncoder* CommandEncoder::get_command_encoder() {
  if (!encoder_) {
    error_.check();
    encoder_ = NS::RetainPtr(
        buffer_->computeCommandEncoder(MTL::DispatchTypeConcurrent));
    fence_ = NS::TransferPtr(device_.mtl_device()->newFence());
  }
  return encoder_.get();
}

Device::Device() : device_(load_device()), residency_sets_(device_.get()) {
  auto pool = new_scoped_memory_pool();
  default_library_ = NS::TransferPtr(load_default_library(device_.get()));
  arch_ = env::metal_gpu_arch();
  if (arch_.empty()) {
    arch_ = std::string(device_->architecture()->name()->utf8String());
  }
  int ag_tens = 0;
  int ag_ones = 0;
  if (arch_.size() >= 3) {
    ag_tens = arch_[arch_.size() - 3] - '0';
    ag_ones = arch_[arch_.size() - 2] - '0';
    ag_tens = (ag_tens < 10 && ag_tens >= 0) ? ag_tens : 0;
    ag_ones = (ag_ones < 10 && ag_ones >= 0) ? ag_ones : 0;
  }
  arch_gen_ = ag_tens * 10 + ag_ones;
  auto arch = arch_.back();
  switch (arch) {
    case 'p': // phone
      max_ops_per_buffer_ = 20;
      max_mb_per_buffer_ = 40;
      break;
    case 'g': // base, pro
      max_ops_per_buffer_ = 40;
      max_mb_per_buffer_ = 40;
      break;
    case 's': // max
      max_ops_per_buffer_ = 50;
      max_mb_per_buffer_ = 50;
      break;
    case 'd': // ultra
      max_ops_per_buffer_ = 50;
      max_mb_per_buffer_ = 50;
      break;
    default: // default to medium
      max_ops_per_buffer_ = 40;
      max_mb_per_buffer_ = 40;
      break;
  }
  max_ops_per_buffer_ = env::max_ops_per_buffer(max_ops_per_buffer_);
  max_mb_per_buffer_ = env::max_mb_per_buffer(max_mb_per_buffer_);
}

Device::~Device() = default;

MTL::Library* Device::get_library(
    const std::string& name,
    const std::string& path /* = "" */) {
  {
    std::shared_lock rlock(library_mtx_);
    if (auto it = library_map_.find(name); it != library_map_.end()) {
      return it->second.get();
    }
  }

  std::unique_lock wlock(library_mtx_);
  if (auto it = library_map_.find(name); it != library_map_.end()) {
    return it->second.get();
  }

  auto new_lib = load_library(device_.get(), name, path.c_str());
  library_map_.insert({name, NS::TransferPtr(new_lib)});
  return new_lib;
}

NS::SharedPtr<MTL::Library> Device::build_library_(
    const std::string& source_string,
    const CompileOptions& compile_options) {
  auto pool = new_scoped_memory_pool();

  auto ns_code =
      NS::String::string(source_string.c_str(), NS::ASCIIStringEncoding);

  NS::Error* error = nullptr;
  auto options = MTL::CompileOptions::alloc()->init()->autorelease();
  set_compile_options(options, compile_options);
  options->setLanguageVersion(get_metal_version());
#ifndef NDEBUG
  if (options->languageVersion() >= MTL::LanguageVersion3_2) {
    options->setEnableLogging(true);
  }
#endif
  auto mtl_lib = NS::TransferPtr(device_->newLibrary(ns_code, options, &error));

  // Throw error if unable to compile library
  if (!mtl_lib) {
    std::ostringstream msg;
    msg << "[metal::Device] Unable to build metal library from source\n";
    if (error) {
      msg << error->localizedDescription()->utf8String() << "\n";
    }
    throw std::runtime_error(msg.str());
  }

  return mtl_lib;
}

NS::SharedPtr<MTL::Function> Device::get_function_(
    const std::string& name,
    MTL::Library* mtl_lib) {
  auto pool = new_scoped_memory_pool();
  // Pull kernel from library
  auto ns_name = NS::String::string(name.c_str(), NS::ASCIIStringEncoding);
  return NS::TransferPtr(mtl_lib->newFunction(ns_name));
}

NS::SharedPtr<MTL::Function> Device::get_function_(
    const std::string& name,
    const std::string& specialized_name,
    const MTLFCList& func_consts,
    MTL::Library* mtl_lib) {
  if (func_consts.empty() && (specialized_name == name)) {
    return get_function_(name, mtl_lib);
  }

  auto pool = new_scoped_memory_pool();

  // Prepare function constants
  auto mtl_func_consts =
      MTL::FunctionConstantValues::alloc()->init()->autorelease();

  for (auto [value, type, index] : func_consts) {
    mtl_func_consts->setConstantValue(value, type, index);
  }

  // Prepare function desc
  auto desc = MTL::FunctionDescriptor::functionDescriptor();
  desc->setName(NS::String::string(name.c_str(), NS::ASCIIStringEncoding));
  desc->setSpecializedName(
      NS::String::string(specialized_name.c_str(), NS::ASCIIStringEncoding));
  desc->setConstantValues(mtl_func_consts);

  // Pull kernel from library
  NS::Error* error = nullptr;
  auto mtl_function = NS::TransferPtr(mtl_lib->newFunction(desc, &error));

  // Throw error if unable to build metal function
  if (!mtl_function) {
    std::ostringstream msg;
    msg << "[metal::Device] Unable to load function " << name << "\n";
    if (error) {
      msg << error->localizedDescription()->utf8String() << "\n";
    }
    throw std::runtime_error(msg.str());
  }

  return mtl_function;
}

NS::SharedPtr<MTL::ComputePipelineState> Device::get_kernel_(
    const std::string& name,
    const MTL::Function* mtl_function) {
  // Compile kernel to compute pipeline
  NS::Error* error = nullptr;
  NS::SharedPtr<MTL::ComputePipelineState> kernel;

  if (mtl_function) {
    kernel =
        NS::TransferPtr(device_->newComputePipelineState(mtl_function, &error));
  }

  // Throw error if unable to compile metal function
  if (!mtl_function || !kernel) {
    std::ostringstream msg;
    msg << "[metal::Device] Unable to load kernel " << name << "\n";
    if (error) {
      msg << error->localizedDescription()->utf8String() << "\n";
    }
    throw std::runtime_error(msg.str());
  }

  return kernel;
}

NS::SharedPtr<MTL::ComputePipelineState> Device::get_kernel_(
    const std::string& name,
    const MTL::Function* mtl_function,
    const MTL::LinkedFunctions* linked_functions) {
  // Check inputs
  if (!linked_functions) {
    return get_kernel_(name, mtl_function);
  }

  if (!mtl_function) {
    std::ostringstream msg;
    msg << "[metal::Device] Unable to load kernel " << name << "\n";
    throw std::runtime_error(msg.str());
  }

  auto pool = new_scoped_memory_pool();

  // Prepare compute pipeline state descriptor
  auto desc = MTL::ComputePipelineDescriptor::alloc()->init()->autorelease();
  desc->setComputeFunction(mtl_function);
  desc->setLinkedFunctions(linked_functions);

  // Compile kernel to compute pipeline
  NS::Error* error = nullptr;
  auto kernel = NS::TransferPtr(device_->newComputePipelineState(
      desc, MTL::PipelineOptionNone, nullptr, &error));

  // Throw error if unable to compile metal function
  if (!kernel) {
    std::ostringstream msg;
    msg << "[metal::Device] Unable to load kernel " << name << "\n";
    if (error) {
      msg << error->localizedDescription()->utf8String() << "\n";
    }
    throw std::runtime_error(msg.str());
  }

  return kernel;
}

MTL::Library* Device::get_library(
    const std::string& name,
    const CompileOptions& compile_options,
    const std::function<std::string(void)>& builder) {
  {
    std::shared_lock rlock(library_mtx_);
    if (auto it = library_map_.find(name); it != library_map_.end()) {
      return it->second.get();
    }
  }

  std::unique_lock wlock(library_mtx_);
  if (auto it = library_map_.find(name); it != library_map_.end()) {
    return it->second.get();
  }

  auto mtl_lib = build_library_(builder(), compile_options);
  library_map_.insert({name, mtl_lib});
  return mtl_lib.get();
}

void Device::clear_library(const std::string& name) {
  std::unique_lock wlock(library_mtx_);
  if (auto it = library_map_.find(name); it != library_map_.end()) {
    library_kernels_.erase(it->second.get());
    library_map_.erase(it);
  }
}

NS::SharedPtr<MTL::LinkedFunctions> Device::get_linked_functions_(
    const std::vector<MTL::Function*>& funcs) {
  if (funcs.empty()) {
    return nullptr;
  }

  auto pool = new_scoped_memory_pool();
  auto lfuncs = NS::TransferPtr(MTL::LinkedFunctions::linkedFunctions());
  NS::Array* funcs_arr = NS::Array::array(
      reinterpret_cast<const NS::Object* const*>(funcs.data()), funcs.size());
  lfuncs->setPrivateFunctions(funcs_arr);
  return lfuncs;
}

MTL::ComputePipelineState* Device::get_kernel_(
    const std::string& base_name,
    MTL::Library* mtl_lib,
    const std::string& hash_name,
    const MTLFCList& func_consts /* = {} */,
    const std::vector<MTL::Function*>& linked_functions /* = {} */) {
  // Single writer allowed
  std::unique_lock wlock(kernel_mtx_);

  // Try loading again to avoid loading twice
  auto& kernel_map_ = library_kernels_[mtl_lib];
  if (auto it = kernel_map_.find(hash_name); it != kernel_map_.end()) {
    return it->second.get();
  }

  auto pool = new_scoped_memory_pool();

  // Pull kernel from library
  auto mtl_function = get_function_(base_name, hash_name, func_consts, mtl_lib);

  // Compile kernel to compute pipeline
  auto mtl_linked_funcs = get_linked_functions_(linked_functions);
  auto kernel =
      get_kernel_(hash_name, mtl_function.get(), mtl_linked_funcs.get());

  // Add kernel to cache
  kernel_map_.insert({hash_name, kernel});
  phone_prof_register(kernel.get(), hash_name);  // phone-store GPU profiler: kernel names

  return kernel.get();
}

MTL::ComputePipelineState* Device::get_kernel(
    const std::string& base_name,
    MTL::Library* mtl_lib,
    const std::string& hash_name /* = "" */,
    const MTLFCList& func_consts /* = {} */,
    const std::vector<MTL::Function*>& linked_functions /* = {} */) {
  const auto& kname = hash_name.empty() ? base_name : hash_name;
  {
    // Multiple readers allowed
    std::shared_lock lock(kernel_mtx_);

    // Look for cached kernel
    auto library_it = library_kernels_.find(mtl_lib);
    if (library_it != library_kernels_.end()) {
      auto kernel_it = library_it->second.find(kname);
      if (kernel_it != library_it->second.end()) {
        return kernel_it->second.get();
      }
    }
  }
  return get_kernel_(base_name, mtl_lib, kname, func_consts, linked_functions);
}

MTL::ComputePipelineState* Device::get_kernel(
    const std::string& base_name,
    const std::string& hash_name /*  = "" */,
    const MTLFCList& func_consts /*  = {} */,
    const std::vector<MTL::Function*>& linked_functions /*  = {} */) {
  return get_kernel(
      base_name,
      default_library_.get(),
      hash_name,
      func_consts,
      linked_functions);
}

Device& device(mlx::core::Device) {
  // Leak singleton device intentionally, to avoid cases where a compute kernel
  // returns and tries to access the object after it has been freed by the main
  // thread teardown.
  static Device* metal_device = new Device;
  return *metal_device;
}

CommandEncoder& get_command_encoder(Stream s) {
  auto& encoders = get_command_encoders();
  auto it = encoders.find(s.index);
  if (it == encoders.end()) {
    auto& global_encoders = get_global_command_encoders();
    it = global_encoders.find(s.index);
    if (it == global_encoders.end()) {
      throw std::runtime_error(
          fmt::format(
              "There is no Stream(gpu, {}) in current thread.", s.index));
    }
  }
  return it->second;
}

std::unordered_map<int, CommandEncoder>& get_command_encoders() {
  if (is_main_thread()) {
    static auto* encoders = new std::unordered_map<int, CommandEncoder>();
    return *encoders;
  } else {
    static thread_local std::unordered_map<int, CommandEncoder> encoders;
    return encoders;
  }
}

std::unordered_map<int, CommandEncoder>& get_global_command_encoders() {
  static std::unordered_map<int, CommandEncoder> encoders;
  return encoders;
}

NS::SharedPtr<NS::AutoreleasePool> new_scoped_memory_pool() {
  return NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
}

bool is_nax_available() {
#ifdef MLX_METAL_NO_NAX
  return false;
#else
  auto _check_nax = []() {
    bool can_use_nax = false;
    if (__builtin_available(
            macOS 26.2, iOS 26.2, tvOS 26.2, visionOS 26.2, *)) {
      can_use_nax = true;
    }
    auto& d = metal::device(mlx::core::Device::gpu);
    auto arch = d.get_architecture().back();
    auto gen = d.get_architecture_gen();
    can_use_nax &= gen >= (arch == 'p' ? 18 : 17);
    return can_use_nax;
  };
  static bool is_nax_available_ = _check_nax();
  return is_nax_available_;
#endif
}

} // namespace mlx::core::metal

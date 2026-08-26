# ANE reverse-engineering notes and protocol

- Status: first static and behavioral inventory complete; dynamic ABI probes pending
- Host profiled: Mac Studio with M3 Ultra and 256 GB unified memory
- OS profiled: macOS 26.5.2, build 25F84
- ANE compiler framework: 9.509.0
- Date: 2026-08-26

> This page records machine fingerprint and recovered ABI. For where the whole ANE question stands today, and which of its conclusions have since been corrected, see `docs/ANE_STATUS.md`.

## Conclusion

The initial assumption is partly right. Core ML presents a generalized public
model and prediction interface, but that public layer does not expose the
compiler target, program format, physical ANE selection, memory mapping, queue,
or synchronization controls needed by this project. Those layers are private
and are specialized by the installed OS for the local ANE architecture.

Core ML is still the right guide. It gives us:

- Apple's public model schema and MIL operation semantics;
- a compiler oracle for accepted shapes, layouts, dtypes, and placement;
- a numerical oracle for equivalent public and direct programs;
- a controlled way to observe which representation changes at each layer.

It does not give us a public ANE backend API. The correct approach is
differential reverse engineering: compile and run the same small graph through
public Core ML and the private in-memory path, change one property at a time,
and record artifacts, placement, errors, numerical output, and timing. OMLX and
maderix provide working hypotheses; this machine decides which remain true.

## Installed stack

The local system exposes this pipeline:

```text
Core ML model specification / ML Program
                  |
          public CoreML.framework
                  |
       CoreML compiler and segmenter
        /                         \
 CoreMLOdie / E5RT            Espresso
        \                         /
             MIL representation
                  |
       ANECompilerService.xpc
                  |
      ANECompiler.framework
                  |
    AppleNeuralEngine.framework
                  |
       aned / aneuserd daemon
                  |
           driver + firmware
                  |
            physical ANE(s)
```

The framework binaries are supplied through the dyld shared cache rather than
as ordinary files. Their Info plists and Objective-C runtime surfaces remain
inspectable. The installed frameworks include:

| Layer | Local component | What it establishes |
|---|---|---|
| Public API | `CoreML.framework` | model compilation, loading, prediction, compute plan |
| New segmentation | `CoreMLOdie.framework` | E5/BNNS program segmentation services |
| IR | `MIL.framework` | MIL implementation, largely C++ rather than Objective-C |
| Legacy/runtime graph | `Espresso.framework` | neural-network graph, shapes, weights, ANE planning |
| Device compiler | `ANECompiler.framework` and `ANECompilerService.xpc` | installed compiler specialization |
| Runtime client | `AppleNeuralEngine.framework` | program lifecycle, requests, IOSurface mapping |
| Execution service | `/usr/libexec/aned`, `/usr/libexec/aneuserd` | daemon and driver boundary |

The compiler service is an Apple platform process with access to the protected
ANE model cache. A client supplies model inputs and options; Apple owns the
hardware-specific compiled artifacts and cache lifecycle.

## Machine fingerprint

The private `_ANEDeviceInfo` class reports:

```text
hasANE                 true
numANEs                2
numANECores            32
aneArchitectureType    h15g
aneSubType             h15
aneSubTypeVariant      g
aneSubTypeAndVariant   h15d
buildVersion           25F84
```

The architecture strings are opaque identifiers, not a public compatibility
contract. They prove that runtime behavior must be keyed by more than the
marketing name "M3 Ultra." Any compiled-program cache or capability decision
must include at least the OS build, ANE architecture identifiers, compiler
version, MIL hash, weight hash, and execution options.

### Numeric fingerprint — h15g, measured 2026-08-26

The identifiers above name the part; they do not say how it computes. This is
the arithmetic fingerprint, established by `Tools/ANE/numerics.m` with probes
whose answers are exact integers differing between candidate hardware models,
so each one discriminates rather than estimates. Bryngelson 2026 measures
M1/H13 and M5/H17 and names A15/M3 the one unmeasured rail; this is that rail.

| property | M1/H13 (published) | **h15g (measured here)** | probe |
|---|---|---|---|
| wide fp32-class accumulator | yes | **yes** | `ones x4096` returns 4096; all 16 cancellation triples survive to 16384 |
| first-stage tile-of-four fp16 rounding | yes, returns 5116 | **no — returns 5120 exact** | `4096 + 1024 ones` |
| per-product fp16 rounding | yes | **yes** | residual probe returns exact 0 |
| partial saturation at 2^15 | yes | **yes, and returns ZERO** | `triple@32768` returns 0, not 16 |
| denormals inside the MAC | flushed | **flushed — the dominant error source in practice** | `denormals x16` returns 0 |
| tie rounding | to even (2048) | **up (2050)** — like M5 | `2048 + 1` |

The pipeline this establishes:

```text
fp16 in -> fp16 multiply, PRODUCT ROUNDED TO FP16
        -> wide (fp32-class) accumulator, exact
        -> fp16 out, ties round up
```

**Four consequences for H3.**

**Denormal flush dominates everything else when operands are small.** fp16's
smallest normal is 6.10e-5, and any product below it is dropped inside the
multiply-accumulate. The QKV spike measured 3.17% relative error against an
FP64 oracle purely because a third of its products underflowed; scaling both
operands so the products clear the boundary drops that to 2.07e-4, a **152x**
improvement, and lands on the arithmetic floor the model below predicts. The
GPU's error over the same sweep is invariant to seven figures, which is what
identifies the effect as an absolute floor rather than rounding.

The practical consequence is that the quantity to measure on real activations
is the **fraction of products below 6.10e-5**, per projection — not the
cancellation ratio. And the mitigation is cheap: a linear projection is
homogeneous, so a power-of-two scale applied to the operands and undone on the
output is exact in fp16 and moves the distribution off the boundary. This is
FP16 training's loss-scaling, and it must be tried before concluding that FP16
breaches contract 8.


The accumulator is exact, so a long contraction is not itself a hazard. Drift
comes entirely from per-product rounding, which makes the relative error
`2^-12 * cancellation / sqrt(K)` — a property of the data's conditioning, not
of K. That model reproduces the differential harness's measured 1.35% at
K=5376 to within 4%, and it means **fc2 at K=14,336 is the least
error-sensitive of the four projections, not the most**.

The accumulator width is a fixed hardware property rather than a per-program
setting, which predicts — and explains — the static and dynamic paths measuring
bit-identical. There is no better-behaved lowering to go looking for.

The saturation row is the one that can lose a render silently. A dot product
whose running partial reaches 2^15 returns **zero**, not `inf` and not `NaN`,
so nothing downstream can detect it. Cancellation-heavy projections run
partials far above their own result, so the quantity that must be bounded is
`max|interior partial|`, not `max|output|`. Any production gate needs that
bound per projection before a shape is enabled.

### Telemetry surface

`Tools/ANE/counters.h` reads whole-engine telemetry through IOReport
(`/usr/lib/libIOReport.dylib`) with no entitlement. The per-task-descriptor
counters remain gated — forcing the stats mask non-zero turns a successful load
into a rejected one.

| channel | on this host | use |
|---|---|---|
| `Energy Model \| ANE0_0` / `ANE0_1` | **yes**, per die | work signal; a die that did nothing reads exactly 0 mJ |
| `SoC Stats \| Cluster Power States \| DIE_*_ANE0` | **yes**, states `ACT`/`INACT` | DVFS residency; 0% is conclusive, >0% is not |
| `SoC Stats \| Events \| *_ANE_*TRG` | **yes** | throttle triggers; a non-zero count invalidates the arm |
| `PMP0/1 \| DCS BW \| AMCC0/1 RD,WR` | **yes**, 32 bins of 64 GB/s | total DRAM bandwidth per die; peak bin and time above the floor are the honest readings |
| `AMC Stats \| Perf Counters \| ANE0 RD/WR` | **absent** | engine DRAM traffic is not measurable on M3 Ultra |

Per-die energy is what makes this worth having: the wall clock cannot say which
die ran a job, and energy can. That is how the instance-hint result in
the telemetry below was found.

Two matching traps, both hit while building this. `"ANE"` is a substring of
*Miscellaneous*, *VLane* and *LanesEng*, so a loose channel filter silently
pulls in network and storage counters — match group and subgroup exactly. And
the power states are named `ACT` and `INACT`, where `ACT` is a substring of
`INACT`, so a contains-test files every idle tick as work.

## What Core ML revealed

A public one-layer `64 x 64`, `S=64` convolution-as-GEMM was compiled using
`coremlcompiler`. The compiled model remained a device-neutral Espresso bundle:

```text
model.espresso.net
model.espresso.shape
model.espresso.weights
coremldata.bin
metadata.json
```

The network file preserves the semantic convolution and the shape file records
`[N,C,H,W] = [1,64,1,64]`. It does not contain the final ANE program. Loading
the model with `.cpuAndNeuralEngine` causes later device specialization, and
`MLComputePlan` reports the convolution's preferred device as the Neural Engine.

This separates two questions that must not be conflated:

1. Can Core ML represent and place the operation on ANE?
2. What private MIL/compiler/runtime contract produces the program actually
   executed by this silicon and OS build?

For H3, the public ceiling probe answers the first question. The direct-runtime
experiments must answer the second.

## Runtime ABI inventory

`Tools/ANE/inspect-runtime.m` loads a framework without constructing a model or
submitting work. It records Objective-C classes, superclass relationships,
instance sizes, selectors, and type encodings. Build and run it with:

```bash
xcrun clang -fobjc-arc -framework Foundation -ldl \
  Tools/ANE/inspect-runtime.m -o /tmp/h3-inspect-runtime
/tmp/h3-inspect-runtime > /tmp/h3-ane-runtime.txt
```

On build 25F84 the AppleNeuralEngine image exposes 34 Objective-C classes and
the following relevant surface.

### Model description and lifecycle

`_ANEInMemoryModelDescriptor` accepts:

```text
+ modelWithMILText:weights:optionsPlist:
```

Its identifier is derived from network text, weights, and the options plist.
`_ANEInMemoryModel` exposes:

```text
+ inMemoryModelWithDescriptor:
- compileWithQoS:options:error:
- loadWithQoS:options:error:
- evaluateWithQoS:options:request:error:
- unloadWithQoS:error:
- compiledModelExists
- hexStringIdentifier
- mapIOSurfacesWithRequest:cacheInference:error:
- unmapIOSurfacesWithRequest:
- programHandle
- intermediateBufferHandle
- queueDepth / setQueueDepth:
```

This confirms distinct compile, load, mapping, and evaluation phases. It also
confirms that program and intermediate-buffer handles exist below Core ML.

### Requests and buffers

`_ANERequest` exposes inputs and outputs independently from a `weightsBuffer`:

```text
+ requestWithInputs:inputIndices:outputs:outputIndices:
    weightsBuffer:perfStats:procedureIndex:
```

It also has variants carrying shared events and a transaction handle.
`_ANEIOSurfaceObject` supports a start offset and no-retain construction, while
`_ANEBuffer` couples an IOSurface object with a symbol index and a source value.
The mapper's recovered type encoding contains an array of 128 buffer records;
that is evidence about the current mapping structure, not yet proof of a
supported 128-buffer API limit.

The differential pass resolved this lead. `weightsBuffer` was accepted by the
request and evaluation APIs but substituting a different weight matrix did not
change the output. Two ordinary tensor inputs did work: changing the independent
weight IOSurface changed the next evaluation without recompilation and matched
the CPU oracle. Use the two-input program as the baseline; the layout and rate consequences are
under *MIL and weight representation* below.

The installed runtime names that entitlement explicitly as
`com.apple.aned.private.adapterWeight.allow`. That strongly suggests the field
belongs to Apple's adapter-weight facility and will be refused to an ordinary
CLI. Probe it once to capture the actual error; do not design around it unless
the unentitled path succeeds. The ordinary tensor-input and packed-input paths
remain the likely deployable options.

### Synchronization

The runtime exposes:

```text
_ANESharedEvents
_ANESharedWaitEvent
_ANESharedSignalEvent
_ANEChainingRequest
_ANEInputBuffersReady
_ANEOutputSetEnqueue
```

This is evidence of driver-level wait and signal support, and on build 25F84
none of it turned out to be needed.

`evaluateWithQoS:` blocks the calling thread, so overlap is obtained by keeping
the GPU busy across that block rather than by signalling into it: submit the
GPU's share with `MLX.asyncEval` on its own stream, then call the engine.
Measured 20.2 ms overlapped against 39.9 ms serial, for work that is 19.2 and
19.8 ms alone.

Two related seam questions, both answered: `MLXFast.metalKernel` allocates its
own output buffer and **cannot** be pointed at an IOSurface, while
`MLXArray(rawPointer:)` **does** map one at the same address. So results come
back by adoption and inputs go out by copy — and the copy is cheap, because
materialising the activation for the engine costs nothing beyond work MLX had to
do anyway (1.214x from a materialised input against 1.216x from a lazy chain).

### Instrumentation

`_ANEPerformanceStats` exposes hardware execution time, performance counter
data, raw ANE statistics, and a driver-mask conversion. This should become the
primary ANE timing source if it can be enabled without privileged access.
Wall-clock time remains necessary to price submission, mapping, and boundary
overhead.

### Physical instances

The runtime reports two physical ANEs. OMLX selects them with these private
execution option keys:

```text
kANEFProcedureVariantHint = 1
kANEFAneInstanceHint      = 1 or 2
```

**`kANEFAneInstanceHint` does not select a die.** Probed with per-die IOReport
energy (`Energy Model|ANE0_0` and `ANE0_1`), a job submitted with the hint set
to 2 burned its energy on die 0 and left die 1 powered down:

| arm | die 0 | die 1 |
|---|---|---|
| ANE hint 1, alone | 1204 mJ | 0 mJ |
| **ANE hint 2, alone** | **1209 mJ** | **0 mJ** |
| two evaluations in flight | 1201 mJ | 1214 mJ |

What engages the second die is **concurrency**: two evaluations outstanding at
once, which the kernel load-balancer then spreads. Measured at the production
shard, one evaluation is 19.90 ms and two concurrent are 19.91 — the second die
is free. Two sequential are 39.7. `h3_ane_run_pair` exists for exactly this
reason and runs its two shards on separate threads.

Pass the hint anyway, since it is harmless, but never rely on it for placement
or for per-die weight residency — the latter would be a correctness hazard.

## MIL and weight representation

Textual MIL 1.3 goes to the in-memory descriptor. `H3ANEBridge` expresses a
linear as a matmul over two runtime tensor inputs:

```text
a: [1, K, 1, S]   activation, sequence as the minor axis
w: [1, K, 1, N]   weight, contraction axis leading
y: [1, S, 1, N]   output, row-major [S, N] and contiguous
```

reshaping `a` to `[1,1,K,S]`, transposing to `[1,1,S,K]`, and matmul-ing against
`w` reshaped to `[1,1,K,N]`. Three layout facts were measured and none are
guessable:

- **The activation wants the sequence as its minor axis.** Declaring it
  `[1,S,1,K]` and contracting the last axis of both operands via `transpose_y`
  runs at 2.42 TFLOP/s a die and takes 30 seconds to compile; the form above
  runs at 3.85 and compiles in under 100 ms.
- **The output needs no transpose.** Reshaping the matmul result straight to
  `[1,S,1,N]` gives the caller's own orientation, contiguous, so it can be
  adopted into MLX rather than copied.
- **The minor extent must pad well.** 3.87 TFLOP/s a die at S=14336 and 16384,
  but 2.45 at S=15731, which is prime. Round the compiled length up to a
  multiple of 64 and slice the surplus off.

Rows are padded to a 64-byte stride within each surface. Weight binding was
settled by differential test: `weightsBuffer` is accepted by the request API but
substituting a different matrix does not change the output — it belongs to
Apple's adapter-weight facility, gated by
`com.apple.aned.private.adapterWeight.allow`. **Two ordinary tensor inputs are
the primary path**: changing the weight IOSurface changes the next evaluation
with no recompilation, and matches a CPU oracle to 2.2e-4. Static and dynamic
lowerings are bit-identical, so there is no better lowering to find.

Static weights use a `BLOBFILE` value at a 64-byte-aligned chunk. The observed
blob convention includes a 64-byte file header, 64-byte chunk headers, type and
length fields, and payload offsets. This is compatible with the open Core ML
MIL schema's external blob model, but the textual syntax and compiler lowering
remain private implementation details.

Core ML Tools is open source under BSD-3-Clause and publishes the model and MIL
protobuf schemas, converter IR, shape/type semantics, and blob references. It
does not publish the on-device ANE compiler or execution ABI. Use its schema and
passes to generate controlled inputs; do not mistake it for the hardware
backend.

## Resulting design rules

- Do not build a generalized ANE backend. Build a versioned h15g/25F84 backend
  profile plus a capability probe, then add profiles only from evidence.
- Represent activation and weight as separate ordinary inputs, with every FP16
  channel row padded to a 64-byte physical stride.
- Keep private types behind a narrow Objective-C C ABI.
- Generate MIL and blob layouts from declarative test descriptions so every
  compiler experiment is reproducible.
- Treat Core ML as the semantic and placement oracle, not the dispatch layer.
- Test `weightsBuffer` before committing to packed dynamic weights.
- Treat synchronization as a measured driver behavior, not an API promise.
- Preserve raw failures and ABI fingerprints; private API drift is expected.
- Keep the existing MLX path authoritative and unchanged when ANE is disabled.

## Sources

- [Apple Core ML Tools](https://github.com/apple/coremltools) — public converter, model schemas, MIL schemas, and Core ML Python bridge.
- [Core ML MIL protobuf schema](https://github.com/apple/coremltools/blob/main/mlmodel/format/MIL.proto) — public program, function, operation, type, and external blob representation.
- [Core ML model protobuf schema](https://github.com/apple/coremltools/blob/main/mlmodel/format/Model.proto) — public model types and specification versions.
- [OMLX Qwen 3.5 ANE path](https://github.com/jundot/omlx/blob/main/omlx/custom_kernels/qwen35_prefill/csrc/qwen35_ane.mm) — in-memory lifecycle, instance hints, IOSurface/Metal interop, procedure banks, and measured synchronization choices.
- [OMLX ANE prefill notes](https://github.com/jundot/omlx/blob/main/docs/experimental/qwen35_ane_prefill.md) — dual-instance behavior, overlap measurements, and address-window observations.
- [maderix/ANE](https://github.com/maderix/ANE) — minimal private-runtime probes and dynamic-weight MIL experiments.
- [Apple: Deploying Transformers on the Apple Neural Engine](https://machinelearning.apple.com/research/neural-engine-transformers) — public ANE layout and bandwidth guidance.
- [Apple Neural Engine: Architecture, Programming, and Performance](https://arxiv.org/abs/2606.22283) — independent reverse engineering of the compiler, program, driver, firmware, and hardware layers.

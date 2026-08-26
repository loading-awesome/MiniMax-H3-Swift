// overlap.mm — Metal/ANE zero-copy visibility and coexecution probe.
//
// Build from the repository root; this embeds the differential harness helpers
// so the ABI and MIL generator stay identical to the numerical tests.

#define DIFF_K 5376
#define DIFF_N 4096
#define DIFF_S 2048
#define DIFF_ITERATIONS 1
#define DIFF_DYNAMIC_ONLY 1
#define main h3_embedded_differential_main
#include "differential.m"
#undef main

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <dispatch/dispatch.h>
#include <algorithm>
#include <vector>
#include <string>
#include <utility>
#include "counters.h"

// Every arm is bracketed by the engine's own telemetry, because the wall clock
// cannot say which die ran. "Both ANEs completed in 23.9 ms" is equally
// consistent with two dies working and with one die doing both jobs back to
// back while the other idled — the residency counters separate those, and the
// throttle counters say whether the comparison was taken across a clock event.
static ANECounters Counters;
static std::vector<std::pair<std::string, ANECounterDelta>> ArmTelemetry;

// Variadic: an arm body is ordinary statements containing commas and braces,
// which a two-parameter macro would split on.
#define MEASURED_ARM(label, ...) do { \
        ANECountersBegin(&Counters); __VA_ARGS__; \
        ArmTelemetry.push_back(std::make_pair(std::string(label), \
                                              ANECountersEnd(&Counters))); \
    } while (0)

static const int GPUDimension = 4096;
static const int Samples = 15;

static IOSurfaceRef NewTensorSurface(int channels, int width) {
    size_t rowBytes = RowBytes(width);
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth: @(width),
        (id)kIOSurfaceHeight: @(channels),
        (id)kIOSurfaceBytesPerElement: @2,
        (id)kIOSurfaceBytesPerRow: @(rowBytes),
        (id)kIOSurfaceAllocSize: @(rowBytes * channels),
        (id)kIOSurfacePixelFormat: @0,
    });
}

static double Median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    size_t middle = values.size() / 2;
    return values.size() % 2 ? values[middle] : (values[middle-1] + values[middle]) * 0.5;
}

static id<MTLComputePipelineState> FillPipeline(id<MTLDevice> device) {
    NSString *source = @"#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "kernel void fill_one(texture2d<half, access::write> out [[texture(0)]], "
        "uint2 gid [[thread_position_in_grid]]) {\n"
        "  if (gid.x < out.get_width() && gid.y < out.get_height()) out.write(half(1.0), gid);\n"
        "}\n";
    NSError *error = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
    if (!library) { fprintf(stderr,"Metal fill library failed: %s\n",error.description.UTF8String); return nil; }
    id<MTLFunction> function = [library newFunctionWithName:@"fill_one"];
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
    if (!pipeline) fprintf(stderr,"Metal fill pipeline failed: %s\n",error.description.UTF8String);
    return pipeline;
}

static BOOL FillActivationWithMetal(id<MTLDevice> device, id<MTLCommandQueue> queue,
                                    IOSurfaceRef surface) {
    MTLTextureDescriptor *descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR16Float
        width:S height:K mipmapped:NO];
    descriptor.usage = MTLTextureUsageShaderWrite | MTLTextureUsageShaderRead;
    descriptor.storageMode = MTLStorageModeShared;
    id<MTLTexture> texture = [device newTextureWithDescriptor:descriptor iosurface:surface plane:0];
    if (!texture) { fprintf(stderr,"Metal could not create an R16Float view of the ANE IOSurface\n"); return NO; }
    id<MTLComputePipelineState> pipeline = FillPipeline(device);
    if (!pipeline) return NO;
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:pipeline]; [encoder setTexture:texture atIndex:0];
    MTLSize threads = MTLSizeMake(16,16,1);
    [encoder dispatchThreads:MTLSizeMake(S,K,1) threadsPerThreadgroup:threads];
    [encoder endEncoding]; [command commit]; [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted) {
        fprintf(stderr,"Metal activation fill failed: %s\n",command.error.description.UTF8String); return NO;
    }
    return YES;
}

struct GPUWork {
    id<MTLCommandQueue> queue;
    MPSMatrixMultiplication *gemm;
    MPSMatrix *a;
    MPSMatrix *b;
    MPSMatrix *c;
};

static GPUWork NewGPUWork(id<MTLDevice> device) {
    size_t rowBytes = ((size_t)GPUDimension * 2 + 255) & ~(size_t)255;
    size_t bytes = rowBytes * GPUDimension;
    id<MTLBuffer> aBuffer = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> bBuffer = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> cBuffer = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    memset(aBuffer.contents,0x34,bytes); memset(bBuffer.contents,0x31,bytes); memset(cBuffer.contents,0,bytes);
    MPSMatrixDescriptor *d = [MPSMatrixDescriptor matrixDescriptorWithRows:GPUDimension
        columns:GPUDimension rowBytes:rowBytes dataType:MPSDataTypeFloat16];
    GPUWork work;
    work.queue = [device newCommandQueue];
    work.a = [[MPSMatrix alloc] initWithBuffer:aBuffer descriptor:d];
    work.b = [[MPSMatrix alloc] initWithBuffer:bBuffer descriptor:d];
    work.c = [[MPSMatrix alloc] initWithBuffer:cBuffer descriptor:d];
    work.gemm = [[MPSMatrixMultiplication alloc] initWithDevice:device transposeLeft:NO transposeRight:NO
        resultRows:GPUDimension resultColumns:GPUDimension interiorColumns:GPUDimension alpha:1.0 beta:0.0];
    return work;
}

static id<MTLCommandBuffer> EncodeGPU(GPUWork &work) {
    id<MTLCommandBuffer> command = [work.queue commandBuffer];
    [work.gemm encodeToCommandBuffer:command leftMatrix:work.a rightMatrix:work.b resultMatrix:work.c];
    return command;
}

static double RunGPU(GPUWork &work) {
    id<MTLCommandBuffer> command = EncodeGPU(work);
    uint64_t begin=mach_absolute_time(); [command commit]; [command waitUntilCompleted];
    if(command.status!=MTLCommandBufferStatusCompleted) return NAN;
    return Milliseconds(mach_absolute_time()-begin);
}

static double RunANE(id model, IOSurfaceRef activation, IOSurfaceRef weight, IOSurfaceRef output,
                     NSDictionary *options) {
    uint64_t begin=mach_absolute_time();
    BOOL ok=EvaluateWithOptions(model,@[(__bridge id)activation,(__bridge id)weight],output,NULL,@0,options,NULL);
    return ok?Milliseconds(mach_absolute_time()-begin):NAN;
}

int main(void) {
    @autoreleasepool {
        setbuf(stdout,NULL); mach_timebase_info(&Timebase);
        dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",RTLD_NOW);
        DescriptorClass=NSClassFromString(@"_ANEInMemoryModelDescriptor"); ModelClass=NSClassFromString(@"_ANEInMemoryModel");
        RequestClass=NSClassFromString(@"_ANERequest"); SurfaceObjectClass=NSClassFromString(@"_ANEIOSurfaceObject");
        NSDictionary *options1=@{@"kANEFProcedureVariantHint":@1,@"kANEFAneInstanceHint":@1};
        NSDictionary *options2=@{@"kANEFProcedureVariantHint":@1,@"kANEFAneInstanceHint":@2};
        ExecutionOptions=options1;
        id<MTLDevice> device=MTLCreateSystemDefaultDevice(); id<MTLCommandQueue> fillQueue=[device newCommandQueue];
        if(!device||!DescriptorClass){fprintf(stderr,"Required Metal/ANE runtime unavailable\n");return 2;}

        BuiltModel model=Build(SeparateMIL(),@{});
        if(!model.loaded){fprintf(stderr,"ANE build failed: %s\n",model.error.UTF8String);Destroy(model);return 2;}
        ExecutionOptions=options2; BuiltModel model2=Build(SeparateMIL(),@{});
        if(!model2.loaded){fprintf(stderr,"ANE instance 2 build failed: %s\n",model2.error.UTF8String);Destroy(model);Destroy(model2);return 2;}
        IOSurfaceRef activation=NewTensorSurface(K,S),weight=NewTensorSurface(K,N),output=NewTensorSurface(N,S),output2=NewTensorSurface(N,S);
        if(!activation||!weight||!output||!output2){fprintf(stderr,"IOSurface allocation failed\n");Destroy(model);Destroy(model2);return 2;}

        _Float16 *identity=( _Float16 *)calloc((size_t)K*N,2);
        for(int i=0;i<N&&i<K;++i)identity[(size_t)i*N+i]=(_Float16)1.0f;
        WriteTensor(weight,identity,K,N); free(identity);
        if(!FillActivationWithMetal(device,fillQueue,activation)){Destroy(model);return 2;}

        double visibilityMS=RunANE(model.model,activation,weight,output,options1);
        IOSurfaceLock(output,kIOSurfaceLockReadOnly,NULL); const _Float16 *first=(const _Float16 *)IOSurfaceGetBaseAddress(output);
        float maxError=0; size_t outStride=RowBytes(S)/2;
        for(int n=0;n<N;++n)for(int s=0;s<S;++s)maxError=fmaxf(maxError,fabsf((float)first[(size_t)n*outStride+s]-1.0f));
        IOSurfaceUnlock(output,kIOSurfaceLockReadOnly,NULL);
        printf("zero_copy_visibility=%s max_abs=%.6g ane_ms=%.3f\n",maxError<0.001f?"PASS":"FAIL",maxError,visibilityMS);
        if(maxError>=0.001f){Destroy(model);return 3;}

        Counters=ANECountersOpen();
        printf("ane_telemetry=%s\n",Counters.available?"IOReport":"unavailable");
        GPUWork gpu=NewGPUWork(device);
        RunGPU(gpu); RunANE(model.model,activation,weight,output,options1); RunANE(model2.model,activation,weight,output2,options2);
        std::vector<double> gpuOnly,aneOnly,overlapWall,overlapANE,overlapGPU;
        MEASURED_ARM("gpu isolated", for(int i=0;i<Samples;++i)gpuOnly.push_back(RunGPU(gpu)));
        MEASURED_ARM("ane1 isolated", for(int i=0;i<Samples;++i)aneOnly.push_back(RunANE(model.model,activation,weight,output,options1)));
        MEASURED_ARM("gpu + ane1",
        for(int i=0;i<Samples;++i){
            __block double aneMS=NAN; dispatch_group_t group=dispatch_group_create();
            dispatch_semaphore_t start=dispatch_semaphore_create(0);
            dispatch_group_async(group,dispatch_get_global_queue(QOS_CLASS_USER_INITIATED,0),^{
                @autoreleasepool { dispatch_semaphore_wait(start,DISPATCH_TIME_FOREVER); aneMS=RunANE(model.model,activation,weight,output,options1); }
            });
            id<MTLCommandBuffer> command=EncodeGPU(gpu); uint64_t begin=mach_absolute_time();
            dispatch_semaphore_signal(start); [command commit]; [command waitUntilCompleted]; dispatch_group_wait(group,DISPATCH_TIME_FOREVER);
            overlapWall.push_back(Milliseconds(mach_absolute_time()-begin)); overlapANE.push_back(aneMS);
            overlapGPU.push_back((command.GPUEndTime-command.GPUStartTime)*1000.0);
        });
        double g=Median(gpuOnly),a=Median(aneOnly),wall=Median(overlapWall),aConcurrent=Median(overlapANE),gConcurrent=Median(overlapGPU);
        double ideal=fmax(g,a),serial=g+a;
        printf("gpu_isolated_ms=%.3f ane_isolated_ms=%.3f\n",g,a);
        printf("overlap_wall_ms=%.3f ane_during_overlap_ms=%.3f gpu_hardware_during_overlap_ms=%.3f\n",wall,aConcurrent,gConcurrent);
        printf("ideal_parallel_ms=%.3f serialized_ms=%.3f overlap_efficiency=%.1f%% speedup_vs_serial=%.3fx\n",
               ideal,serial,100.0*(serial-wall)/(serial-ideal),serial/wall);
        printf("gpu_shape=%dx%dx%d ane_shape=S%d_K%d_N%d ane_instance=1 samples=%d\n",
               GPUDimension,GPUDimension,GPUDimension,S,K,N,Samples);
        double gpuTF=2.0*GPUDimension*GPUDimension*GPUDimension/g/1e9;
        double aneTF=2.0*S*K*N/a/1e9;
        printf("gpu_isolated_tflops=%.2f ane_isolated_tflops=%.2f\n",gpuTF,aneTF);

        std::vector<double> ane2Only,dualWall,tripleWall;
        MEASURED_ARM("ane2 isolated", for(int i=0;i<Samples;++i)ane2Only.push_back(RunANE(model2.model,activation,weight,output2,options2)));
        MEASURED_ARM("ane1 + ane2",
        for(int i=0;i<Samples;++i){
            __block double a1=NAN,a2=NAN;dispatch_group_t group=dispatch_group_create();dispatch_semaphore_t start=dispatch_semaphore_create(0);
            dispatch_group_async(group,dispatch_get_global_queue(QOS_CLASS_USER_INITIATED,0),^{@autoreleasepool{dispatch_semaphore_wait(start,DISPATCH_TIME_FOREVER);a1=RunANE(model.model,activation,weight,output,options1);}});
            dispatch_group_async(group,dispatch_get_global_queue(QOS_CLASS_USER_INITIATED,0),^{@autoreleasepool{dispatch_semaphore_wait(start,DISPATCH_TIME_FOREVER);a2=RunANE(model2.model,activation,weight,output2,options2);}});
            uint64_t begin=mach_absolute_time();dispatch_semaphore_signal(start);dispatch_semaphore_signal(start);dispatch_group_wait(group,DISPATCH_TIME_FOREVER);
            dualWall.push_back(Milliseconds(mach_absolute_time()-begin));(void)a1;(void)a2;
        });
        MEASURED_ARM("gpu + ane1 + ane2",
        for(int i=0;i<Samples;++i){
            __block double a1=NAN,a2=NAN;dispatch_group_t group=dispatch_group_create();dispatch_semaphore_t start=dispatch_semaphore_create(0);
            dispatch_group_async(group,dispatch_get_global_queue(QOS_CLASS_USER_INITIATED,0),^{@autoreleasepool{dispatch_semaphore_wait(start,DISPATCH_TIME_FOREVER);a1=RunANE(model.model,activation,weight,output,options1);}});
            dispatch_group_async(group,dispatch_get_global_queue(QOS_CLASS_USER_INITIATED,0),^{@autoreleasepool{dispatch_semaphore_wait(start,DISPATCH_TIME_FOREVER);a2=RunANE(model2.model,activation,weight,output2,options2);}});
            id<MTLCommandBuffer> command=EncodeGPU(gpu);uint64_t begin=mach_absolute_time();dispatch_semaphore_signal(start);dispatch_semaphore_signal(start);[command commit];[command waitUntilCompleted];dispatch_group_wait(group,DISPATCH_TIME_FOREVER);
            tripleWall.push_back(Milliseconds(mach_absolute_time()-begin));(void)a1;(void)a2;
        });
        double a2=Median(ane2Only),dual=Median(dualWall),triple=Median(tripleWall);
        printf("ane2_isolated_ms=%.3f dual_ane_wall_ms=%.3f dual_speedup_vs_serial=%.3fx\n",a2,dual,(a+a2)/dual);
        printf("gpu_plus_two_ane_wall_ms=%.3f three_way_speedup_vs_serial=%.3fx ideal_parallel_ms=%.3f\n",
               triple,(g+a+a2)/triple,fmax(g,fmax(a,a2)));
        if(Counters.available){
            printf("\nengine telemetry per arm (IOReport; DRAM byte counters are absent on this host)\n");
            for(size_t i=0;i<ArmTelemetry.size();++i)
                ANECountersReport(ArmTelemetry[i].first.c_str(),ArmTelemetry[i].second);
            printf("  energy is the work signal: a die that did nothing reads exactly 0 mJ,\n"
                   "  which the wall clock cannot tell you. clock-up is DVFS residency and\n"
                   "  lags work by an idle timeout, so 0%% is conclusive and >0%% is not.\n"
                   "  a non-zero throttle count invalidates the arm it appears in.\n");
        }
        CFRelease(activation);CFRelease(weight);CFRelease(output);CFRelease(output2);Destroy(model);Destroy(model2);
    }
    return 0;
}

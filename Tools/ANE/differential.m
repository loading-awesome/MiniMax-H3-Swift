// differential.m — Behavioral ANE weight-path probe.
//
// This intentionally uses private AppleNeuralEngine.framework interfaces. It is
// a research tool, not code intended for product distribution.

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <objc/message.h>
#import <objc/runtime.h>
#import <dlfcn.h>
#import <mach/mach_time.h>
#include <math.h>

#ifndef DIFF_K
#define DIFF_K 64
#endif
#ifndef DIFF_N
#define DIFF_N 64
#endif
#ifndef DIFF_S
#define DIFF_S 64
#endif
#ifndef DIFF_ITERATIONS
#define DIFF_ITERATIONS 40
#endif
#ifndef DIFF_ANE_INSTANCE
#define DIFF_ANE_INSTANCE 0
#endif
enum { K = DIFF_K, N = DIFF_N, S = DIFF_S, ITERATIONS = DIFF_ITERATIONS };

typedef struct {
    BOOL compiled;
    BOOL loaded;
    id model;
    NSString *directory;
    NSString *error;
} BuiltModel;

typedef struct {
    float maxAbs;
    float relativeRMS;
    float cosine;
} Metrics;

static Class DescriptorClass;
static Class ModelClass;
static Class RequestClass;
static Class SurfaceObjectClass;
static mach_timebase_info_data_t Timebase;
static unsigned BuildSerial;
static NSDictionary *ExecutionOptions;

static NSString *MILHeader(void) {
    return @"program(1.3)\n"
            "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
            "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
            "{\"coremltools-version\", \"9.0\"}})]\n{\n";
}

static NSString *PackedMIL(void) {
    NSMutableString *m = [NSMutableString stringWithString:MILHeader()];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1,%d,1,%d]> packed) {\n", K, S + N];
    [m appendFormat:@"        tensor<int32,[4]> ba=const()[name=string(\"ba\"),val=tensor<int32,[4]>([0,0,0,0])];\n"
                     "        tensor<int32,[4]> sa=const()[name=string(\"sa\"),val=tensor<int32,[4]>([1,%d,1,%d])];\n"
                     "        tensor<fp16,[1,%d,1,%d]> a=slice_by_size(x=packed,begin=ba,size=sa)[name=string(\"a\")];\n", K, S, K, S];
    [m appendFormat:@"        tensor<int32,[4]> bw=const()[name=string(\"bw\"),val=tensor<int32,[4]>([0,0,0,%d])];\n"
                     "        tensor<int32,[4]> sw=const()[name=string(\"sw\"),val=tensor<int32,[4]>([1,%d,1,%d])];\n"
                     "        tensor<fp16,[1,%d,1,%d]> w=slice_by_size(x=packed,begin=bw,size=sw)[name=string(\"w\")];\n", S, K, N, K, N];
    [m appendFormat:@"        tensor<int32,[4]> ra=const()[name=string(\"ra\"),val=tensor<int32,[4]>([1,1,%d,%d])];\n"
                     "        tensor<fp16,[1,1,%d,%d]> a2=reshape(shape=ra,x=a)[name=string(\"a2\")];\n"
                     "        tensor<int32,[4]> pm=const()[name=string(\"pm\"),val=tensor<int32,[4]>([0,1,3,2])];\n"
                     "        tensor<fp16,[1,1,%d,%d]> at=transpose(perm=pm,x=a2)[name=string(\"at\")];\n", K, S, K, S, S, K];
    [m appendFormat:@"        tensor<int32,[4]> rw=const()[name=string(\"rw\"),val=tensor<int32,[4]>([1,1,%d,%d])];\n"
                     "        tensor<fp16,[1,1,%d,%d]> w2=reshape(shape=rw,x=w)[name=string(\"w2\")];\n"
                     "        bool bf=const()[name=string(\"bf\"),val=bool(false)];\n"
                     "        tensor<fp16,[1,1,%d,%d]> mm=matmul(transpose_x=bf,transpose_y=bf,x=at,y=w2)[name=string(\"mm\")];\n"
                     "        tensor<fp16,[1,1,%d,%d]> mt=transpose(perm=pm,x=mm)[name=string(\"mt\")];\n"
                     "        tensor<int32,[4]> ro=const()[name=string(\"ro\"),val=tensor<int32,[4]>([1,%d,1,%d])];\n"
                     "        tensor<fp16,[1,%d,1,%d]> y=reshape(shape=ro,x=mt)[name=string(\"y\")];\n"
                     "    } -> (y);\n}\n", K, N, K, N, S, N, N, S, N, S, N, S];
    return m;
}

static NSString *SeparateMIL(void) {
    NSMutableString *m = [NSMutableString stringWithString:MILHeader()];
    [m appendFormat:@"    func main<ios18>(tensor<fp16,[1,%d,1,%d]> a, tensor<fp16,[1,%d,1,%d]> w) {\n", K, S, K, N];
    [m appendFormat:@"        tensor<int32,[4]> ra=const()[name=string(\"ra\"),val=tensor<int32,[4]>([1,1,%d,%d])];\n"
                     "        tensor<fp16,[1,1,%d,%d]> a2=reshape(shape=ra,x=a)[name=string(\"a2\")];\n"
                     "        tensor<int32,[4]> pm=const()[name=string(\"pm\"),val=tensor<int32,[4]>([0,1,3,2])];\n"
                     "        tensor<fp16,[1,1,%d,%d]> at=transpose(perm=pm,x=a2)[name=string(\"at\")];\n", K, S, K, S, S, K];
    [m appendFormat:@"        tensor<int32,[4]> rw=const()[name=string(\"rw\"),val=tensor<int32,[4]>([1,1,%d,%d])];\n"
                     "        tensor<fp16,[1,1,%d,%d]> w2=reshape(shape=rw,x=w)[name=string(\"w2\")];\n"
                     "        bool bf=const()[name=string(\"bf\"),val=bool(false)];\n"
                     "        tensor<fp16,[1,1,%d,%d]> mm=matmul(transpose_x=bf,transpose_y=bf,x=at,y=w2)[name=string(\"mm\")];\n"
                     "        tensor<fp16,[1,1,%d,%d]> mt=transpose(perm=pm,x=mm)[name=string(\"mt\")];\n"
                     "        tensor<int32,[4]> ro=const()[name=string(\"ro\"),val=tensor<int32,[4]>([1,%d,1,%d])];\n"
                     "        tensor<fp16,[1,%d,1,%d]> y=reshape(shape=ro,x=mt)[name=string(\"y\")];\n"
                     "    } -> (y);\n}\n", K, N, K, N, S, N, N, S, N, S, N, S];
    return m;
}

static NSData *WeightBlob(const _Float16 *weightsKN) {
    const size_t payloadBytes = N * K * sizeof(_Float16);
    NSMutableData *data = [NSMutableData dataWithLength:128 + payloadBytes];
    uint8_t *p = (uint8_t *)data.mutableBytes;
    p[0] = 1; p[4] = 2;
    *(uint32_t *)(p + 64) = 0xDEADBEEF;
    *(uint32_t *)(p + 68) = 1;
    *(uint64_t *)(p + 72) = payloadBytes;
    *(uint64_t *)(p + 80) = 128;
    _Float16 *dst = (_Float16 *)(p + 128);
    for (int n = 0; n < N; ++n)
        for (int k = 0; k < K; ++k)
            dst[n * K + k] = weightsKN[k * N + n];
    return data;
}

static NSString *StaticMIL(void) {
    NSMutableString *m = [NSMutableString stringWithString:MILHeader()];
    [m appendFormat:@"    func main<ios18>(tensor<fp16,[1,%d,1,%d]> x) {\n", K, S];
    [m appendString:@"        string pt=const()[name=string(\"pt\"),val=string(\"valid\")];\n"
                     "        tensor<int32,[2]> st=const()[name=string(\"st\"),val=tensor<int32,[2]>([1,1])];\n"
                     "        tensor<int32,[4]> pd=const()[name=string(\"pd\"),val=tensor<int32,[4]>([0,0,0,0])];\n"
                     "        tensor<int32,[2]> dl=const()[name=string(\"dl\"),val=tensor<int32,[2]>([1,1])];\n"
                     "        int32 gr=const()[name=string(\"gr\"),val=int32(1)];\n"];
    [m appendFormat:@"        tensor<fp16,[%d,%d,1,1]> w=const()[name=string(\"w\"),val=tensor<fp16,[%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/weight.bin\"),offset=uint64(64)))];\n"
                     "        tensor<fp16,[1,%d,1,%d]> y=conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=w,x=x)[name=string(\"y\")];\n"
                     "    } -> (y);\n}\n", N, K, N, K, N, S];
    return m;
}

static IOSurfaceRef NewSurface(size_t bytes) {
    size_t allocation = (bytes + 16383) & ~(size_t)16383;
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth: @(allocation), (id)kIOSurfaceHeight: @1,
        (id)kIOSurfaceBytesPerElement: @1, (id)kIOSurfaceBytesPerRow: @(allocation),
        (id)kIOSurfaceAllocSize: @(allocation), (id)kIOSurfacePixelFormat: @0
    });
}

static size_t RowBytes(int width) { return ((size_t)width * 2 + 63) & ~(size_t)63; }
static size_t TensorBytes(int channels, int width) { return (size_t)channels * RowBytes(width); }

static void WriteSurface(IOSurfaceRef surface, const void *bytes, size_t count) {
    IOSurfaceLock(surface, 0, NULL);
    memcpy(IOSurfaceGetBaseAddress(surface), bytes, count);
    IOSurfaceUnlock(surface, 0, NULL);
}

static void WriteTensor(IOSurfaceRef surface, const _Float16 *logical, int channels, int width) {
    const size_t rowBytes = RowBytes(width);
    IOSurfaceLock(surface, 0, NULL);
    uint8_t *base = (uint8_t *)IOSurfaceGetBaseAddress(surface);
    memset(base, 0, TensorBytes(channels, width));
    for (int c = 0; c < channels; ++c)
        memcpy(base + c * rowBytes, logical + c * width, (size_t)width * 2);
    IOSurfaceUnlock(surface, 0, NULL);
}

static void ReadTensor(IOSurfaceRef surface, _Float16 *logical, int channels, int width) {
    const size_t rowBytes = RowBytes(width);
    IOSurfaceLock(surface, kIOSurfaceLockReadOnly, NULL);
    const uint8_t *base = (const uint8_t *)IOSurfaceGetBaseAddress(surface);
    for (int c = 0; c < channels; ++c)
        memcpy(logical + c * width, base + c * rowBytes, (size_t)width * 2);
    IOSurfaceUnlock(surface, kIOSurfaceLockReadOnly, NULL);
}

static BuiltModel Build(NSString *mil, NSDictionary *weights) {
    BuiltModel result = {0};
    NSData *milData = [mil dataUsingEncoding:NSUTF8StringEncoding];
    id descriptor = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(DescriptorClass,
        @selector(modelWithMILText:weights:optionsPlist:), milData, weights, nil);
    result.model = ((id(*)(Class,SEL,id))objc_msgSend)(ModelClass,
        @selector(inMemoryModelWithDescriptor:), descriptor);
    if (!result.model) { result.error = @"model construction returned nil"; return result; }

    NSString *identifier = ((id(*)(id,SEL))objc_msgSend)(result.model, @selector(hexStringIdentifier));
    result.directory = [NSTemporaryDirectory() stringByAppendingPathComponent:
        [NSString stringWithFormat:@"h3-ane-diff-%d-%u-%@", getpid(), BuildSerial++, identifier]];
    NSFileManager *fm = NSFileManager.defaultManager;
    NSError *error = nil;
    [fm createDirectoryAtPath:[result.directory stringByAppendingPathComponent:@"weights"]
        withIntermediateDirectories:YES attributes:nil error:&error];
    [milData writeToFile:[result.directory stringByAppendingPathComponent:@"model.mil"] options:0 error:&error];
    [weights enumerateKeysAndObjectsUsingBlock:^(NSString *path, NSDictionary *entry, BOOL *stop) {
        (void)stop;
        NSData *data = entry[@"data"];
        NSString *relative = [path stringByReplacingOccurrencesOfString:@"@model_path/" withString:@""];
        NSString *destination = [result.directory stringByAppendingPathComponent:relative];
        [fm createDirectoryAtPath:destination.stringByDeletingLastPathComponent
            withIntermediateDirectories:YES attributes:nil error:nil];
        [data writeToFile:destination atomically:YES];
    }];
    if ([result.model respondsToSelector:@selector(setModelURL:)]) {
        ((void(*)(id,SEL,id))objc_msgSend)(result.model, @selector(setModelURL:),
            [NSURL fileURLWithPath:result.directory]);
    }
    error = nil;
    result.compiled = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(result.model,
        @selector(compileWithQoS:options:error:), 21, ExecutionOptions, &error);
    if (!result.compiled) { result.error = error.description; return result; }
    error = nil;
    result.loaded = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(result.model,
        @selector(loadWithQoS:options:error:), 21, ExecutionOptions, &error);
    if (!result.loaded) result.error = error.description;
    return result;
}

static void Destroy(BuiltModel model) {
    NSError *error = nil;
    if (model.loaded) ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(model.model,
        @selector(unloadWithQoS:error:), 21, &error);
    if (model.directory) [NSFileManager.defaultManager removeItemAtPath:model.directory error:nil];
}

static BOOL EvaluateWithOptions(id model, NSArray *inputs, IOSurfaceRef output, IOSurfaceRef weightsBuffer,
                                NSNumber *procedureIndex, NSDictionary *options, NSString **errorText) {
    NSMutableArray *objects = [NSMutableArray array];
    NSMutableArray *indices = [NSMutableArray array];
    for (NSUInteger i = 0; i < inputs.count; ++i) {
        IOSurfaceRef surface = (__bridge IOSurfaceRef)inputs[i];
        id object = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(SurfaceObjectClass,
            @selector(objectWithIOSurface:), surface);
        [objects addObject:object]; [indices addObject:@(i)];
    }
    id outObject = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(SurfaceObjectClass,
        @selector(objectWithIOSurface:), output);
    id weightObject = weightsBuffer ? ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(SurfaceObjectClass,
        @selector(objectWithIOSurface:), weightsBuffer) : nil;
    id request = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(RequestClass,
        @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
        objects, indices, @[outObject], @[@0], weightObject, nil, procedureIndex ?: @0);
    NSError *error = nil;
    BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(model,
        @selector(evaluateWithQoS:options:request:error:), 21, options, request, &error);
    if (!ok && errorText) *errorText = error.description ?: @"unknown evaluation error";
    return ok;
}

static BOOL Evaluate(id model, NSArray *inputs, IOSurfaceRef output, IOSurfaceRef weightsBuffer,
                     NSNumber *procedureIndex, NSString **errorText) {
    return EvaluateWithOptions(model,inputs,output,weightsBuffer,procedureIndex,ExecutionOptions,errorText);
}

static void FillInputs(_Float16 *x, _Float16 *w1, _Float16 *w2) {
    for (int k = 0; k < K; ++k) for (int s = 0; s < S; ++s) {
        float v = sinf((float)(k * 17 + s * 13 + 3) * 0.071f) * 0.35f;
        x[k * S + s] = (_Float16)v;
    }
    for (int k = 0; k < K; ++k) for (int n = 0; n < N; ++n) {
        float v = cosf((float)(k * 11 + n * 19 + 5) * 0.053f) * 0.12f;
        w1[k * N + n] = (_Float16)v;
        w2[k * N + n] = (_Float16)(v * -0.625f + ((k == n) ? 0.2f : 0.0f));
    }
}

static void Reference(const _Float16 *x, const _Float16 *w, _Float16 *y) {
    for (int n = 0; n < N; ++n) for (int s = 0; s < S; ++s) {
        float sum = 0;
        for (int k = 0; k < K; ++k) sum += (float)x[k * S + s] * (float)w[k * N + n];
        y[n * S + s] = (_Float16)sum;
    }
}

static Metrics Compare(const _Float16 *actual, const _Float16 *expected) {
    double error2 = 0, expected2 = 0, actual2 = 0, dot = 0;
    float maxAbs = 0;
    for (int i = 0; i < N * S; ++i) {
        double a = (float)actual[i], e = (float)expected[i], d = a - e;
        maxAbs = fmaxf(maxAbs, fabsf((float)d));
        error2 += d*d; expected2 += e*e; actual2 += a*a; dot += a*e;
    }
    Metrics m = { maxAbs, (float)sqrt(error2 / fmax(expected2, 1e-30)),
                  (float)(dot / sqrt(fmax(actual2 * expected2, 1e-30))) };
    return m;
}

static double Milliseconds(uint64_t ticks) {
    return (double)ticks * Timebase.numer / Timebase.denom / 1e6;
}

static void PrintResult(NSString *name, BOOL ok, NSString *error, Metrics m1, Metrics m2,
                        BOOL mutationObserved, double medianMS) {
    printf("%-15s compile/load/eval=%s", name.UTF8String, ok ? "OK" : "FAIL");
    if (!ok) { printf(" error=%s\n", (error ?: @"unknown").UTF8String); return; }
    printf("  W1[max=%.6g rrms=%.6g cos=%.8f]", m1.maxAbs, m1.relativeRMS, m1.cosine);
    printf("  W2[max=%.6g rrms=%.6g cos=%.8f]", m2.maxAbs, m2.relativeRMS, m2.cosine);
    printf("  mutation=%s  eval_ms=%.4f\n", mutationObserved ? "YES" : "NO", medianMS);
}

static double TimedEvaluations(id model, NSArray *inputs, IOSurfaceRef output) {
    uint64_t samples[ITERATIONS];
    for (int i = 0; i < ITERATIONS; ++i) {
        uint64_t begin = mach_absolute_time();
        if (!Evaluate(model, inputs, output, NULL, @0, NULL)) return NAN;
        samples[i] = mach_absolute_time() - begin;
    }
    for (int i = 0; i < ITERATIONS; ++i) for (int j = i + 1; j < ITERATIONS; ++j)
        if (samples[j] < samples[i]) { uint64_t t=samples[i]; samples[i]=samples[j]; samples[j]=t; }
    return Milliseconds(samples[ITERATIONS / 2]);
}

static void RunDynamic(NSString *name, NSString *mil, BOOL packed,
                       const _Float16 *x, const _Float16 *w1, const _Float16 *w2,
                       const _Float16 *ref1, const _Float16 *ref2) {
    BuiltModel model = Build(mil, @{});
    if (!model.loaded) { PrintResult(name, NO, model.error, (Metrics){0}, (Metrics){0}, NO, NAN); Destroy(model); return; }
    IOSurfaceRef xSurface = NULL, wSurface = NULL;
    if (packed) {
        _Float16 *p = (_Float16 *)calloc(K * (S + N), sizeof(_Float16));
        for (int k=0;k<K;++k) { memcpy(p+k*(S+N), x+k*S, S*2); memcpy(p+k*(S+N)+S, w1+k*N, N*2); }
        xSurface = NewSurface(TensorBytes(K,S+N)); WriteTensor(xSurface,p,K,S+N); free(p);
    } else {
        xSurface = NewSurface(TensorBytes(K,S)); wSurface = NewSurface(TensorBytes(K,N));
        WriteTensor(xSurface,x,K,S); WriteTensor(wSurface,w1,K,N);
    }
    IOSurfaceRef out = NewSurface(TensorBytes(N,S));
    _Float16 *got1=(_Float16 *)calloc(N*S,2), *got2=(_Float16 *)calloc(N*S,2);
    NSArray *inputs = packed ? @[(__bridge id)xSurface] : @[(__bridge id)xSurface, (__bridge id)wSurface];
    NSString *error = nil;
    BOOL ok = Evaluate(model.model, inputs, out, NULL, @0, &error); if (ok) ReadTensor(out,got1,N,S);
    if (packed) {
        _Float16 *p=(_Float16 *)calloc(K*(S+N),2); for(int k=0;k<K;++k){memcpy(p+k*(S+N),x+k*S,S*2);memcpy(p+k*(S+N)+S,w2+k*N,N*2);} WriteTensor(xSurface,p,K,S+N);free(p);
    } else WriteTensor(wSurface,w2,K,N);
    if (ok) ok = Evaluate(model.model, inputs, out, NULL, @0, &error); if (ok) ReadTensor(out,got2,N,S);
    Metrics m1=Compare(got1,ref1), m2=Compare(got2,ref2), delta=Compare(got2,got1);
    double ms = ok ? TimedEvaluations(model.model, inputs, out) : NAN;
    PrintResult(name, ok, error, m1, m2, delta.maxAbs > 0.01f, ms);
    free(got1);free(got2);CFRelease(xSurface);if(wSurface)CFRelease(wSurface);CFRelease(out);Destroy(model);
}

static void RunStatic(const _Float16 *x, const _Float16 *w1, const _Float16 *w2,
                      const _Float16 *ref1, const _Float16 *ref2) {
    NSData *blob=WeightBlob(w1);
    BuiltModel model=Build(StaticMIL(), @{@"@model_path/weights/weight.bin":@{@"offset":@0,@"data":blob}});
    if(!model.loaded){PrintResult(@"static+wb",NO,model.error,(Metrics){0},(Metrics){0},NO,NAN);Destroy(model);return;}
    IOSurfaceRef in=NewSurface(TensorBytes(K,S)),out=NewSurface(TensorBytes(N,S)),wb=NewSurface(K*N*2);
    WriteTensor(in,x,K,S);WriteSurface(wb,w2,K*N*2);
    _Float16 *got1=(_Float16 *)calloc(N*S,2),*got2=(_Float16 *)calloc(N*S,2),*got15=(_Float16 *)calloc(N*S,2);NSString *error=nil;
    BOOL ok=Evaluate(model.model,@[(__bridge id)in],out,NULL,@0,&error);if(ok)ReadTensor(out,got1,N,S);
    if(ok)ok=Evaluate(model.model,@[(__bridge id)in],out,wb,@0,&error);if(ok)ReadTensor(out,got2,N,S);
    Metrics m1=Compare(got1,ref1),m2=Compare(got2,ref2),delta=Compare(got2,got1);
    PrintResult(@"static+wb",ok,error,m1,m2,delta.maxAbs>0.01f,ok?TimedEvaluations(model.model,@[(__bridge id)in],out):NAN);
    printf("  note: W2 metrics test whether weightsBuffer overrides the compiled W1 constant.\n");
    BOOL p15=Evaluate(model.model,@[(__bridge id)in],out,NULL,@15,&error);if(p15)ReadTensor(out,got15,N,S);
    Metrics procedureDelta=Compare(got15,got1);
    printf("  procedureIndex 15: %s, delta_from_index_0=%.6g (%s)\n",
           p15?"OK":"FAIL",procedureDelta.maxAbs,procedureDelta.maxAbs==0?"ignored or same procedure":"different output");
    free(got1);free(got2);free(got15);CFRelease(in);CFRelease(out);CFRelease(wb);Destroy(model);
}

int main(void) {
    @autoreleasepool {
        setbuf(stdout,NULL);mach_timebase_info(&Timebase);
        if(!dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",RTLD_NOW)){
            fprintf(stderr,"Unable to load AppleNeuralEngine.framework\n");return 2;
        }
        DescriptorClass=NSClassFromString(@"_ANEInMemoryModelDescriptor");ModelClass=NSClassFromString(@"_ANEInMemoryModel");
        RequestClass=NSClassFromString(@"_ANERequest");SurfaceObjectClass=NSClassFromString(@"_ANEIOSurfaceObject");
        if(!DescriptorClass||!ModelClass||!RequestClass||!SurfaceObjectClass){fprintf(stderr,"Required private classes unavailable\n");return 2;}
        _Float16 *x=(_Float16 *)calloc(K*S,2),*w1=(_Float16 *)calloc(K*N,2),*w2=(_Float16 *)calloc(K*N,2),*r1=(_Float16 *)calloc(N*S,2),*r2=(_Float16 *)calloc(N*S,2);
        FillInputs(x,w1,w2);Reference(x,w1,r1);Reference(x,w2,r2);
        ExecutionOptions = DIFF_ANE_INSTANCE == 0 ? @{} : @{
            @"kANEFProcedureVariantHint": @1, @"kANEFAneInstanceHint": @(DIFF_ANE_INSTANCE)
        };
        printf("ANE differential probe: K=%d N=%d S=%d instance=%d fp16, CPU fp32 accumulation then fp16 rounding\n",K,N,S,DIFF_ANE_INSTANCE);
#ifndef DIFF_DYNAMIC_ONLY
        RunStatic(x,w1,w2,r1,r2);
#endif
        RunDynamic(@"separate-input",SeparateMIL(),NO,x,w1,w2,r1,r2);
        RunDynamic(@"packed-input",PackedMIL(),YES,x,w1,w2,r1,r2);
        free(x);free(w1);free(w2);free(r1);free(r2);
    }
    return 0;
}

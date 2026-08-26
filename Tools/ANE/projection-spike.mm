// projection-spike.mm — One production-shape H3 QKV projection split across
// Metal and both ANE dies. Research-only: it uses private ANE interfaces.

#define DIFF_K 5376
#define DIFF_N 3072
#define DIFF_S 2048
#define DIFF_ITERATIONS 1
#define DIFF_DYNAMIC_ONLY 1
#define main h3_embedded_differential_main
#include "differential.m"
#undef main

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#import <dispatch/dispatch.h>
#include <algorithm>
#include <vector>
#include <fcntl.h>
#include <unistd.h>

enum { TotalN=21504, GPUN=15360, Samples=7 };

static uint16_t FloatToBF16(float value) {
    uint32_t bits; memcpy(&bits,&value,4); bits += 0x7fff + ((bits >> 16) & 1); return (uint16_t)(bits >> 16);
}
static float BF16ToFloat(uint16_t value) { uint32_t bits=(uint32_t)value<<16;float f;memcpy(&f,&bits,4);return f; }
static double Median(std::vector<double> v){std::sort(v.begin(),v.end());return v[v.size()/2];}
static uint32_t Mix(uint32_t x){x^=x>>16;x*=0x7feb352dU;x^=x>>15;x*=0x846ca68bU;x^=x>>16;return x;}
static float Unit(uint32_t x){return ((float)(Mix(x)&0xffffU)-32768.0f)/32768.0f;}
// Both operands scale together, so products scale by the square. This is the
// discriminating knob for the error's mechanism: per-product fp16 rounding is
// relative and therefore scale-invariant, while flushing denormal products to
// zero is an absolute floor and must weaken as the operands grow away from
// fp16's smallest normal (6.10e-5). H3_SPIKE_SCALE=1 is the original fixture.
static float FixtureScale(void){
    static float scale=0;
    if(scale==0){ const char *v=getenv("H3_SPIKE_SCALE"); scale=v?atof(v):1.0f; if(scale<=0)scale=1.0f; }
    return scale;
}
static float ActivationValue(int s,int k){return Unit((uint32_t)s*0x9e3779b9U+(uint32_t)k)*0.03f*FixtureScale();}
static float WeightValue(int k,int n){return Unit((uint32_t)k*0x85ebca6bU+(uint32_t)n+0x1234567U)*0.02f*FixtureScale();}

static IOSurfaceRef NewTensorSurface(int channels,int width){size_t rb=RowBytes(width);return IOSurfaceCreate((__bridge CFDictionaryRef)@{
    (id)kIOSurfaceWidth:@(width),(id)kIOSurfaceHeight:@(channels),(id)kIOSurfaceBytesPerElement:@2,
    (id)kIOSurfaceBytesPerRow:@(rb),(id)kIOSurfaceAllocSize:@(rb*channels),(id)kIOSurfacePixelFormat:@0});}

static id<MTLTexture> Texture(id<MTLDevice> device,IOSurfaceRef surface,int width,int height,MTLTextureUsage usage){
    MTLTextureDescriptor *d=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR16Float width:width height:height mipmapped:NO];
    d.storageMode=MTLStorageModeShared;d.usage=usage;return [device newTextureWithDescriptor:d iosurface:surface plane:0];
}

struct Kernels{id<MTLComputePipelineState> activation,weight,join,permute;};
static void Dispatch2D(id<MTLComputeCommandEncoder> e,id<MTLComputePipelineState> p,int width,int height);
static Kernels MakeKernels(id<MTLDevice> device){
    NSString *source=@"#include <metal_stdlib>\nusing namespace metal;\n"
    "float from_bf16(ushort x){return as_type<float>(uint(x)<<16);}\n"
    "ushort to_bf16(float x){uint u=as_type<uint>(x);u+=0x7fff+((u>>16)&1);return ushort(u>>16);}\n"
    "kernel void activation_to_ane(device const ushort* src[[buffer(0)]],texture2d<half,access::write> dst[[texture(0)]],uint2 g[[thread_position_in_grid]]){"
    "if(g.x<dst.get_width()&&g.y<dst.get_height())dst.write(half(from_bf16(src[g.x*5376+g.y])),g);}\n"
    "kernel void weight_to_ane(device const ushort* src[[buffer(0)]],constant uint& start[[buffer(1)]],texture2d<half,access::write> dst[[texture(0)]],uint2 g[[thread_position_in_grid]]){"
    "if(g.x<dst.get_width()&&g.y<dst.get_height())dst.write(half(from_bf16(src[g.y*21504+start+g.x])),g);}\n"
    "kernel void permute_qkv(device const ushort* src[[buffer(0)]],device ushort* dst[[buffer(1)]],uint2 g[[thread_position_in_grid]]){"
    "if(g.x>=21504||g.y>=5376)return;uint q=g.x/(56*128),r=g.x%(56*128),h=r/128,d=r%128;uint sn=(h*3+q)*128+d;dst[g.y*21504+g.x]=src[sn*5376+g.y];}\n"
    "kernel void join_qkv(device const ushort* gpu[[buffer(0)]],texture2d<half,access::read> a1[[texture(0)]],texture2d<half,access::read> a2[[texture(1)]],device ushort* out[[buffer(1)]],uint2 g[[thread_position_in_grid]]){"
    "if(g.x>=21504||g.y>=2048)return;uint i=g.y*21504+g.x;if(g.x<15360)out[i]=gpu[g.y*15360+g.x];"
    "else if(g.x<18432)out[i]=to_bf16(float(a1.read(uint2(g.y,g.x-15360)).x));"
    "else out[i]=to_bf16(float(a2.read(uint2(g.y,g.x-18432)).x));}\n";
    NSError *e=nil;id<MTLLibrary> lib=[device newLibraryWithSource:source options:nil error:&e];
    if(!lib){fprintf(stderr,"Metal library: %s\n",e.description.UTF8String);return {};}
    Kernels k; k.activation=[device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"activation_to_ane"] error:&e];
    k.weight=[device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"weight_to_ane"] error:&e];
    k.join=[device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"join_qkv"] error:&e];
    k.permute=[device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"permute_qkv"] error:&e];return k;
}

static id<MTLBuffer> LoadCheckpointQKV(id<MTLDevice> device,id<MTLCommandQueue> queue,Kernels kernels,
                                       NSString *path,int block,id<MTLBuffer> transposed){
    int fd=open(path.fileSystemRepresentation,O_RDONLY);if(fd<0){perror("open checkpoint");return nil;}
    uint64_t headerLength=0;if(pread(fd,&headerLength,8,0)!=8){close(fd);return nil;}
    NSMutableData *header=[NSMutableData dataWithLength:(NSUInteger)headerLength];
    if(pread(fd,header.mutableBytes,(size_t)headerLength,8)!=(ssize_t)headerLength){close(fd);return nil;}
    NSError *error=nil;NSDictionary *json=[NSJSONSerialization JSONObjectWithData:header options:0 error:&error];
    NSDictionary *metadata=json[@"__metadata__"];
    if(!metadata[@"repo_id"]&&!metadata[@"partition"]){fprintf(stderr,"checkpoint is not the expected interleaved vendor layout\n");close(fd);return nil;}
    NSString *name=[NSString stringWithFormat:@"blocks.%d.attn.qkv_proj.weight",block];NSDictionary *entry=json[name];
    NSArray *offsets=entry[@"data_offsets"];NSArray *shape=entry[@"shape"];
    if(!entry||![entry[@"dtype"] isEqual:@"BF16"]||shape.count!=2||[shape[0] intValue]!=TotalN||[shape[1] intValue]!=K){
        fprintf(stderr,"checkpoint tensor missing or incompatible: %s\n",name.UTF8String);close(fd);return nil;
    }
    uint64_t begin=[offsets[0] unsignedLongLongValue],end=[offsets[1] unsignedLongLongValue],bytes=end-begin;
    id<MTLBuffer> raw=[device newBufferWithLength:(NSUInteger)bytes options:MTLResourceStorageModeShared];
    if(!raw||pread(fd,raw.contents,(size_t)bytes,(off_t)(8+headerLength+begin))!=(ssize_t)bytes){perror("read qkv");close(fd);return nil;}close(fd);
    id<MTLCommandBuffer> command=[queue commandBuffer];id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
    [encoder setBuffer:raw offset:0 atIndex:0];[encoder setBuffer:transposed offset:0 atIndex:1];Dispatch2D(encoder,kernels.permute,TotalN,K);
    [encoder endEncoding];[command commit];[command waitUntilCompleted];return raw;
}

static void Dispatch2D(id<MTLComputeCommandEncoder> e,id<MTLComputePipelineState> p,int width,int height){
    [e setComputePipelineState:p];[e dispatchThreads:MTLSizeMake(width,height,1) threadsPerThreadgroup:MTLSizeMake(16,16,1)];}

struct GraphGEMM{
    MPSGraph *graph;MPSGraphTensor *x,*w,*y;id<MTLCommandQueue> queue;id<MTLBuffer> xb;id<MTLBuffer> wb;id<MTLBuffer> yb;int n;NSUInteger weightRowBytes;
    GraphGEMM(id<MTLDevice> d,id<MTLBuffer> input,id<MTLBuffer> weight,id<MTLBuffer> output,int outputN,NSUInteger wrb):
      graph([MPSGraph new]),queue([d newCommandQueue]),xb(input),wb(weight),yb(output),n(outputN),weightRowBytes(wrb){
        x=[graph placeholderWithShape:@[@(S),@(K)] dataType:MPSDataTypeBFloat16 name:@"x"];
        w=[graph placeholderWithShape:@[@(K),@(n)] dataType:MPSDataTypeBFloat16 name:@"w"];
        y=[graph matrixMultiplicationWithPrimaryTensor:x secondaryTensor:w name:@"qkv"];
    }
    id<MTLCommandBuffer> encode(){
        id<MTLCommandBuffer> raw=[queue commandBuffer];MPSCommandBuffer *cb=[[MPSCommandBuffer alloc]initWithCommandBuffer:raw];
        MPSGraphTensorData *xd=[[MPSGraphTensorData alloc]initWithMTLBuffer:xb shape:@[@(S),@(K)] dataType:MPSDataTypeBFloat16];
        MPSGraphTensorData *wd=[[MPSGraphTensorData alloc]initWithMTLBuffer:wb shape:@[@(K),@(n)] dataType:MPSDataTypeBFloat16 rowBytes:weightRowBytes];
        MPSGraphTensorData *yd=[[MPSGraphTensorData alloc]initWithMTLBuffer:yb shape:@[@(S),@(n)] dataType:MPSDataTypeBFloat16];
        [graph encodeToCommandBuffer:cb feeds:@{x:xd,w:wd} targetOperations:nil resultsDictionary:@{y:yd} executionDescriptor:nil];
        return cb;
    }
};

static double RunGraph(GraphGEMM &g){id<MTLCommandBuffer> c=g.encode();uint64_t t=mach_absolute_time();[c commit];[c waitUntilCompleted];return Milliseconds(mach_absolute_time()-t);}
static double RunANEProgram(id model,IOSurfaceRef x,IOSurfaceRef w,IOSurfaceRef y,NSDictionary *o){uint64_t t=mach_absolute_time();BOOL ok=EvaluateWithOptions(model,@[(__bridge id)x,(__bridge id)w],y,NULL,@0,o,NULL);return ok?Milliseconds(mach_absolute_time()-t):NAN;}

struct Hybrid{
    id<MTLDevice> device;id<MTLCommandQueue> queue;Kernels kernels;GraphGEMM *gpu;id model1,model2;NSDictionary *o1,*o2;
    id<MTLBuffer> input,joined;IOSurfaceRef ax,w1,w2,y1,y2;id<MTLTexture> at,wt1,wt2,yt1,yt2;
    double lastConvert,lastParallel,lastGPU,lastANE1,lastANE2,lastJoin;
    double run(){
        uint64_t begin=mach_absolute_time(),stage=begin;id<MTLCommandBuffer> convert=[queue commandBuffer];id<MTLComputeCommandEncoder> ce=[convert computeCommandEncoder];
        [ce setBuffer:input offset:0 atIndex:0];[ce setTexture:at atIndex:0];Dispatch2D(ce,kernels.activation,S,K);[ce endEncoding];[convert commit];[convert waitUntilCompleted];
        lastConvert=Milliseconds(mach_absolute_time()-stage);
        __block double a1=NAN,a2=NAN;dispatch_group_t group=dispatch_group_create();dispatch_semaphore_t start=dispatch_semaphore_create(0);
        dispatch_group_async(group,dispatch_get_global_queue(QOS_CLASS_USER_INITIATED,0),^{@autoreleasepool{dispatch_semaphore_wait(start,DISPATCH_TIME_FOREVER);a1=RunANEProgram(model1,ax,w1,y1,o1);}});
        dispatch_group_async(group,dispatch_get_global_queue(QOS_CLASS_USER_INITIATED,0),^{@autoreleasepool{dispatch_semaphore_wait(start,DISPATCH_TIME_FOREVER);a2=RunANEProgram(model2,ax,w2,y2,o2);}});
        id<MTLCommandBuffer> gc=gpu->encode();stage=mach_absolute_time();dispatch_semaphore_signal(start);dispatch_semaphore_signal(start);[gc commit];[gc waitUntilCompleted];dispatch_group_wait(group,DISPATCH_TIME_FOREVER);
        lastParallel=Milliseconds(mach_absolute_time()-stage);lastGPU=(gc.GPUEndTime-gc.GPUStartTime)*1000.0;lastANE1=a1;lastANE2=a2;
        stage=mach_absolute_time();id<MTLCommandBuffer> join=[queue commandBuffer];id<MTLComputeCommandEncoder> je=[join computeCommandEncoder];
        [je setBuffer:gpu->yb offset:0 atIndex:0];[je setBuffer:joined offset:0 atIndex:1];[je setTexture:yt1 atIndex:0];[je setTexture:yt2 atIndex:1];Dispatch2D(je,kernels.join,TotalN,S);[je endEncoding];[join commit];[join waitUntilCompleted];
        lastJoin=Milliseconds(mach_absolute_time()-stage);
        if(isnan(a1)||isnan(a2))return NAN;return Milliseconds(mach_absolute_time()-begin);
    }
};

// An FP64 oracle, because every number this spike reported until now compared
// one approximation against another. The bf16 MPSGraph result was treated as
// truth and the whole difference charged to the engine, which cannot be right:
// the engine's own measured behaviour (Tools/ANE/numerics.m) predicts about
// 0.019% for this fixture, not the 3.17% recorded.
//
// This computes the exact projection in double for a sample of output columns
// drawn from all three shards, then scores the GPU and the raw ANE surfaces
// against it separately. It also reports the fixture's cancellation, since
// relative error is meaningless without it: rel_error ~= 2^-12 * C / sqrt(K).
//
// The ANE side is read from its own fp16 output surface, not from the joined
// bf16 buffer, so the join's bf16 quantisation is not charged to the engine.
struct OracleScore { double maxAbs, relRMS, cancellation, resultRMS; };

static OracleScore ScoreAgainstOracle(const uint16_t *activation,const uint16_t *weightKN,
                                      const int *columns,int columnCount,
                                      const _Float16 *aneShard1,const _Float16 *aneShard2,
                                      const uint16_t *gpuBF16,bool scoreANE){
    double err2=0,ref2=0,cancel=0; double maxAbs=0; size_t count=0;
    for(int c=0;c<columnCount;++c){
        int n=columns[c];
        for(int s=0;s<S;++s){
            double exact=0,absMagnitude=0;
            for(int k=0;k<K;++k){
                double p=(double)BF16ToFloat(activation[(size_t)s*K+k])
                        *(double)BF16ToFloat(weightKN[(size_t)k*TotalN+n]);
                exact+=p; absMagnitude+=fabs(p);
            }
            double got;
            if(scoreANE){
                // y1 covers [GPUN, GPUN+N), y2 covers [GPUN+N, TotalN); both are
                // [N,S] fp16 surfaces indexed by the shard-local channel.
                got = n<GPUN+N ? (double)aneShard1[(size_t)(n-GPUN)*S+s]
                               : (double)aneShard2[(size_t)(n-GPUN-N)*S+s];
            } else {
                got = (double)BF16ToFloat(gpuBF16[(size_t)s*TotalN+n]);
            }
            double d=got-exact;
            err2+=d*d; ref2+=exact*exact; cancel+=absMagnitude;
            maxAbs=fmax(maxAbs,fabs(d)); ++count;
        }
    }
    OracleScore score;
    score.maxAbs=maxAbs;
    score.resultRMS=sqrt(ref2/(double)count);
    score.relRMS=sqrt(err2/fmax(ref2,1e-300));
    score.cancellation=(cancel/(double)count)/fmax(score.resultRMS,1e-300);
    return score;
}

static void Compare(const uint16_t *reference,const uint16_t *actual,int beginN,int endN,const char *name){
    double e2=0,r2=0;float maxe=0;size_t count=0;for(int s=0;s<S;++s)for(int n=beginN;n<endN;++n){size_t i=(size_t)s*TotalN+n;float r=BF16ToFloat(reference[i]),a=BF16ToFloat(actual[i]),d=a-r;e2+=d*d;r2+=r*r;maxe=fmaxf(maxe,fabsf(d));++count;}
    printf("%-12s max_abs=%.6g rel_rms=%.6g elements=%zu\n",name,maxe,sqrt(e2/fmax(r2,1e-30)),count);
}

int main(){@autoreleasepool{
    setbuf(stdout,NULL);mach_timebase_info(&Timebase);dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",RTLD_NOW);
    DescriptorClass=NSClassFromString(@"_ANEInMemoryModelDescriptor");ModelClass=NSClassFromString(@"_ANEInMemoryModel");RequestClass=NSClassFromString(@"_ANERequest");SurfaceObjectClass=NSClassFromString(@"_ANEIOSurfaceObject");
    id<MTLDevice> device=MTLCreateSystemDefaultDevice();id<MTLCommandQueue> queue=[device newCommandQueue];Kernels kernels=MakeKernels(device);
    NSDictionary *o1=@{@"kANEFProcedureVariantHint":@1,@"kANEFAneInstanceHint":@1},*o2=@{@"kANEFProcedureVariantHint":@1,@"kANEFAneInstanceHint":@2};
    ExecutionOptions=o1;BuiltModel bm1=Build(SeparateMIL(),@{});ExecutionOptions=o2;BuiltModel bm2=Build(SeparateMIL(),@{});
    if(!bm1.loaded||!bm2.loaded||!kernels.join){fprintf(stderr,"program construction failed: %s %s\n",bm1.error.UTF8String,bm2.error.UTF8String);return 2;}

    id<MTLBuffer> input=[device newBufferWithLength:(size_t)S*K*2 options:MTLResourceStorageModeShared];
    id<MTLBuffer> weights=[device newBufferWithLength:(size_t)K*TotalN*2 options:MTLResourceStorageModeShared];
    id<MTLBuffer> baselineOut=[device newBufferWithLength:(size_t)S*TotalN*2 options:MTLResourceStorageModeShared];
    id<MTLBuffer> gpuOut=[device newBufferWithLength:(size_t)S*GPUN*2 options:MTLResourceStorageModeShared];
    id<MTLBuffer> joined=[device newBufferWithLength:(size_t)S*TotalN*2 options:MTLResourceStorageModeShared];
    uint16_t *ip=(uint16_t *)input.contents,*wp=(uint16_t *)weights.contents;
    for(int s=0;s<S;++s)for(int k=0;k<K;++k)ip[(size_t)s*K+k]=FloatToBF16(ActivationValue(s,k));
    NSDictionary *environment=NSProcessInfo.processInfo.environment;NSString *checkpoint=environment[@"H3_CHECKPOINT"];
    int checkpointBlock=environment[@"H3_BLOCK"]?[environment[@"H3_BLOCK"] intValue]:0;id<MTLBuffer> rawCheckpoint=nil;
    if(checkpoint.length){rawCheckpoint=LoadCheckpointQKV(device,queue,kernels,checkpoint,checkpointBlock,weights);if(!rawCheckpoint)return 2;}
    else for(int k=0;k<K;++k)for(int n=0;n<TotalN;++n)wp[(size_t)k*TotalN+n]=FloatToBF16(WeightValue(k,n));

    IOSurfaceRef ax=NewTensorSurface(K,S),w1=NewTensorSurface(K,N),w2=NewTensorSurface(K,N),y1=NewTensorSurface(N,S),y2=NewTensorSurface(N,S);
    id<MTLTexture> at=Texture(device,ax,S,K,MTLTextureUsageShaderRead|MTLTextureUsageShaderWrite);
    id<MTLTexture> wt1=Texture(device,w1,N,K,MTLTextureUsageShaderRead|MTLTextureUsageShaderWrite),wt2=Texture(device,w2,N,K,MTLTextureUsageShaderRead|MTLTextureUsageShaderWrite);
    id<MTLTexture> yt1=Texture(device,y1,S,N,MTLTextureUsageShaderRead|MTLTextureUsageShaderWrite),yt2=Texture(device,y2,S,N,MTLTextureUsageShaderRead|MTLTextureUsageShaderWrite);
    id<MTLCommandBuffer> prep=[queue commandBuffer];id<MTLComputeCommandEncoder> pe=[prep computeCommandEncoder];
    uint32_t start1=GPUN,start2=GPUN+N;[pe setBuffer:weights offset:0 atIndex:0];[pe setBytes:&start1 length:4 atIndex:1];[pe setTexture:wt1 atIndex:0];Dispatch2D(pe,kernels.weight,N,K);
    [pe setBytes:&start2 length:4 atIndex:1];[pe setTexture:wt2 atIndex:0];Dispatch2D(pe,kernels.weight,N,K);[pe endEncoding];[prep commit];[prep waitUntilCompleted];

    GraphGEMM baseline(device,input,weights,baselineOut,TotalN,(NSUInteger)TotalN*2);
    GraphGEMM gpu(device,input,weights,gpuOut,GPUN,(NSUInteger)TotalN*2);
    Hybrid hybrid{device,queue,kernels,&gpu,bm1.model,bm2.model,o1,o2,input,joined,ax,w1,w2,y1,y2,at,wt1,wt2,yt1,yt2,0,0,0,0,0,0};
    printf("H3 QKV spike S=%d K=%d N=%d split GPU=%d ANE1=%d ANE2=%d\n",S,K,TotalN,GPUN,N,N);
    printf("weights=%s%s\n",checkpoint.length?checkpoint.UTF8String:"deterministic synthetic",checkpoint.length?[NSString stringWithFormat:@" block=%d",checkpointBlock].UTF8String:"");
    printf("resident bytes: bf16_weight=%.1fMB ane_weights=%.1fMB activation=%.1fMB output=%.1fMB\n",
      weights.length/1048576.0,(TensorBytes(K,N)*2)/1048576.0,input.length/1048576.0,joined.length/1048576.0);
    RunGraph(baseline);hybrid.run();
    std::vector<double> baseTimes,hybridTimes,convertTimes,parallelTimes,gpuTimes,ane1Times,ane2Times,joinTimes;
    for(int i=0;i<Samples;++i){if(i%2==0){baseTimes.push_back(RunGraph(baseline));hybridTimes.push_back(hybrid.run());}else{hybridTimes.push_back(hybrid.run());baseTimes.push_back(RunGraph(baseline));}
      convertTimes.push_back(hybrid.lastConvert);parallelTimes.push_back(hybrid.lastParallel);gpuTimes.push_back(hybrid.lastGPU);ane1Times.push_back(hybrid.lastANE1);ane2Times.push_back(hybrid.lastANE2);joinTimes.push_back(hybrid.lastJoin);}
    uint16_t *ref=(uint16_t *)baselineOut.contents,*got=(uint16_t *)joined.contents;
    Compare(ref,got,0,GPUN,"GPU shard");Compare(ref,got,GPUN,GPUN+N,"ANE shard 1");Compare(ref,got,GPUN+N,TotalN,"ANE shard 2");Compare(ref,got,0,TotalN,"whole QKV");

    {   // Score both processors against an exact reference rather than against
        // each other. Eight columns per shard, every row: enough elements that
        // the RMS is stable, few enough that the FP64 pass costs about a second.
        int gpuCols[8],ane1Cols[8],ane2Cols[8];
        for(int i=0;i<8;++i){
            gpuCols[i]=(GPUN/8)*i+11;
            ane1Cols[i]=GPUN+(N/8)*i+11;
            ane2Cols[i]=GPUN+N+(N/8)*i+11;
        }
        _Float16 *shard1=(_Float16 *)calloc((size_t)N*S,2),*shard2=(_Float16 *)calloc((size_t)N*S,2);
        ReadTensor(y1,shard1,N,S); ReadTensor(y2,shard2,N,S);
        OracleScore gpuScore=ScoreAgainstOracle(ip,wp,gpuCols,8,shard1,shard2,ref,false);
        OracleScore ane1Score=ScoreAgainstOracle(ip,wp,ane1Cols,8,shard1,shard2,ref,true);
        OracleScore ane2Score=ScoreAgainstOracle(ip,wp,ane2Cols,8,shard1,shard2,ref,true);

        printf("\nagainst an FP64 oracle (8 columns per shard, all %d rows)\n",S);
        printf("%-14s %14s %14s %12s\n","processor","max_abs","rel_rms","cancellation");
        printf("%-14s %14.6g %14.6g %12.1fx\n","GPU bf16",gpuScore.maxAbs,gpuScore.relRMS,gpuScore.cancellation);
        printf("%-14s %14.6g %14.6g %12.1fx\n","ANE fp16 s1",ane1Score.maxAbs,ane1Score.relRMS,ane1Score.cancellation);
        printf("%-14s %14.6g %14.6g %12.1fx\n","ANE fp16 s2",ane2Score.maxAbs,ane2Score.relRMS,ane2Score.cancellation);
        double predicted=pow(2.0,-12.0)*ane1Score.cancellation/sqrt((double)K);
        printf("model 2^-12 * C / sqrt(K) predicts rel_rms=%.6g for the ANE shards\n",predicted);
        printf("compare with the ANE-vs-GPU rows above, which charge the engine\n"
               "for the GPU's error as well as its own.\n");
        free(shard1);free(shard2);
    }
    double b=Median(baseTimes),h=Median(hybridTimes);printf("gpu_full_ms=%.3f hybrid_end_to_end_ms=%.3f speedup=%.3fx change=%+.1f%% samples=%d\n",b,h,b/h,100*(h-b)/b,Samples);
    printf("hybrid stages: convert=%.3fms parallel=%.3fms [gpu_hw=%.3f ane1=%.3f ane2=%.3f] join=%.3fms\n",
      Median(convertTimes),Median(parallelTimes),Median(gpuTimes),Median(ane1Times),Median(ane2Times),Median(joinTimes));
    printf("hybrid includes bf16->fp16 activation conversion, two host joins, and fp16->bf16 output join\n");
    CFRelease(ax);CFRelease(w1);CFRelease(w2);CFRelease(y1);CFRelease(y2);Destroy(bm1);Destroy(bm2);
}return 0;}

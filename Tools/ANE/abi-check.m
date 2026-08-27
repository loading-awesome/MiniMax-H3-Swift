// Static ABI check: does every selector H3ANEBridge depends on still exist?
// dlopen + runtime introspection only — no client, no model, no evaluation.
#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <dlfcn.h>
static int fails = 0;
static void chk(const char *cls, const char *sel, bool classMethod) {
    Class c = objc_getClass(cls);
    if (!c) { printf("  MISSING CLASS  %s\n", cls); fails++; return; }
    SEL s = sel_registerName(sel);
    bool ok = classMethod ? class_respondsToSelector(object_getClass(c), s)
                          : class_respondsToSelector(c, s);
    printf("  %-4s %s %s%s\n", ok ? "ok" : "GONE", cls, classMethod ? "+" : "-", sel);
    if (!ok) fails++;
}
int main(void){@autoreleasepool{
    if (!dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_LAZY)) {
        printf("dlopen FAILED: %s\n", dlerror()); return 2;
    }
    printf("AppleNeuralEngine loaded.\n");
    chk("_ANEInMemoryModelDescriptor", "modelWithMILText:weights:optionsPlist:", true);
    chk("_ANEInMemoryModel", "inMemoryModelWithDescriptor:", true);
    chk("_ANERequest", "requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:", true);
    chk("_ANEIOSurfaceObject", "objectWithIOSurface:startOffset:", true);
    chk("_ANEInMemoryModel", "hexStringIdentifier", false);
    chk("_ANEInMemoryModel", "compileWithQoS:options:error:", false);
    chk("_ANEInMemoryModel", "loadWithQoS:options:error:", false);
    chk("_ANEInMemoryModel", "unloadWithQoS:error:", false);
    chk("_ANEInMemoryModel", "evaluateWithQoS:options:request:error:", false);
    chk("_ANEClient", "sharedConnection", true);
    chk("_ANEClient", "beginRealTimeTask", false);
    chk("_ANEClient", "endRealTimeTask", false);
    printf("\n%d selector(s) missing.\n", fails);
    return fails ? 1 : 0;
}}

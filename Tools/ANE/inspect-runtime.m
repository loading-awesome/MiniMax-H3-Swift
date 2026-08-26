// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

// Read-only inventory of the private AppleNeuralEngine Objective-C runtime.
// It loads the framework and prints classes, superclass relationships, instance
// sizes, methods and Objective-C type encodings. It never constructs an ANE
// object or submits work.
//
// Build and run:
//
//   xcrun clang -fobjc-arc -framework Foundation -ldl \
//     Tools/ANE/inspect-runtime.m -o /tmp/h3-inspect-ane
//   /tmp/h3-inspect-ane [framework-binary ...]

#import <Foundation/Foundation.h>
#import <objc/message.h>
#import <objc/runtime.h>
#import <dlfcn.h>

static int compareClasses(const void *lhs, const void *rhs) {
    Class left = *(Class const *)lhs;
    Class right = *(Class const *)rhs;
    return strcmp(class_getName(left), class_getName(right));
}

static void printMethods(Class cls, const char *marker) {
    unsigned int count = 0;
    Method *methods = class_copyMethodList(cls, &count);
    for (unsigned int index = 0; index < count; ++index) {
        Method method = methods[index];
        printf("  %s %s\t%s\n", marker,
               sel_getName(method_getName(method)),
               method_getTypeEncoding(method));
    }
    free(methods);
}

static void printDeviceInfo(void) {
    Class info = NSClassFromString(@"_ANEDeviceInfo");
    if (info == Nil) {
        return;
    }
    id (*objectValue)(Class, SEL) = (id (*)(Class, SEL))objc_msgSend;
    unsigned int (*unsignedValue)(Class, SEL) =
        (unsigned int (*)(Class, SEL))objc_msgSend;
    BOOL (*boolValue)(Class, SEL) = (BOOL (*)(Class, SEL))objc_msgSend;

    printf("device\n");
    if (class_getClassMethod(info, @selector(hasANE)) != NULL) {
        printf("  has-ane\t%s\n",
               boolValue(info, @selector(hasANE)) ? "true" : "false");
    }
    if (class_getClassMethod(info, @selector(numANEs)) != NULL) {
        printf("  instances\t%u\n", unsignedValue(info, @selector(numANEs)));
    }
    if (class_getClassMethod(info, @selector(numANECores)) != NULL) {
        printf("  cores\t%u\n", unsignedValue(info, @selector(numANECores)));
    }
    for (NSString *selectorName in @[
        @"productName", @"buildVersion", @"aneArchitectureType",
        @"aneSubType", @"aneSubTypeVariant", @"aneSubTypeAndVariant"
    ]) {
        SEL selector = NSSelectorFromString(selectorName);
        if (class_getClassMethod(info, selector) == NULL) {
            printf("  %s\t(missing)\n", selectorName.UTF8String);
            continue;
        }
        id value = objectValue(info, selector);
        printf("  %s\t%s\n", selectorName.UTF8String,
               value == nil ? "(nil)" : [[value description] UTF8String]);
    }
}

static void printRuntimeConstants(void) {
    Class strings = NSClassFromString(@"_ANEStrings");
    if (strings == Nil) {
        return;
    }
    id (*objectValue)(Class, SEL) = (id (*)(Class, SEL))objc_msgSend;
    printf("runtime-constants\n");
    for (NSString *selectorName in @[
        @"machServiceName", @"machServiceNamePrivate",
        @"restrictedAccessEntitlement", @"adapterWeightsAccessEntitlement",
        @"compilerServiceAccessEntitlement",
        @"secondaryANECompilerServiceAccessEntitlement",
        @"memoryUnwireAccessEntitlement"
    ]) {
        SEL selector = NSSelectorFromString(selectorName);
        if (class_getClassMethod(strings, selector) == NULL) {
            printf("  %s\t(missing)\n", selectorName.UTF8String);
            continue;
        }
        id value = objectValue(strings, selector);
        printf("  %s\t%s\n", selectorName.UTF8String,
               value == nil ? "(nil)" : [[value description] UTF8String]);
    }
}

static int inspectFramework(const char *path) {
    const char *imageNeedle = strrchr(path, '/');
    imageNeedle = imageNeedle == NULL ? path : imageNeedle + 1;

    void *handle = dlopen(path, RTLD_LAZY | RTLD_LOCAL);
    if (handle == NULL) {
        fprintf(stderr, "dlopen failed for %s: %s\n", path, dlerror());
        return 1;
    }

    int count = objc_getClassList(NULL, 0);
    __unsafe_unretained Class *classes =
        (__unsafe_unretained Class *)calloc((size_t)count, sizeof(Class));
    if (classes == NULL) {
        fprintf(stderr, "class-list allocation failed\n");
        return 2;
    }
    count = objc_getClassList(classes, count);
    qsort(classes, (size_t)count, sizeof(Class), compareClasses);

    printf("framework\t%s\n", path);
    if (strcmp(imageNeedle, "AppleNeuralEngine") == 0) {
        printDeviceInfo();
        printRuntimeConstants();
    }
    for (int index = 0; index < count; ++index) {
        Class cls = classes[index];
        const char *image = class_getImageName(cls);
        const char *imageName = image == NULL ? NULL : strrchr(image, '/');
        imageName = imageName == NULL ? image : imageName + 1;
        if (imageName == NULL || strcmp(imageName, imageNeedle) != 0) {
            continue;
        }

        Class superclass = class_getSuperclass(cls);
        printf("\nclass\t%s\n", class_getName(cls));
        printf("  image\t%s\n", image);
        printf("  superclass\t%s\n",
               superclass == Nil ? "(none)" : class_getName(superclass));
        printf("  instance-size\t%zu\n", class_getInstanceSize(cls));
        printMethods(cls, "-");

        Class metaclass = object_getClass(cls);
        if (metaclass != Nil) {
            printMethods(metaclass, "+");
        }
    }

    free(classes);
    // Objective-C classes remain registered for the process lifetime. Do not
    // dlclose their defining image while the runtime still knows about them.
    (void)handle;
    return 0;
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        const char *defaultPath =
            "/System/Library/PrivateFrameworks/"
            "AppleNeuralEngine.framework/AppleNeuralEngine";
        if (argc == 1) {
            return inspectFramework(defaultPath);
        }
        int result = 0;
        for (int index = 1; index < argc; ++index) {
            int current = inspectFramework(argv[index]);
            if (current != 0) {
                result = current;
            }
        }
        return result;
    }
}

// pair-stress.m — does concurrent dual-die submission fault the ANE's DART?
//
// Isolates one variable. It submits `h3_ane_run_pair` back to back with the
// SAME input and weight surfaces bound to both dies — the worst case, and
// exactly what Tools/ANE/ane-hold.m was doing when the machine panicked with
//
//   sptm_t8110dart_clear_err: dart-ane0: DART instance 1:
//   Unrecoverable secondary error 0x80080008
//
// 72 seconds in. Because it submits continuously the engine never idles, so the
// driver's sleep race is out of the picture and what remains is the mapping
// question: whether one _ANEIOSurfaceObject shared across two concurrent
// requests on two DARTs is legal.
//
// H3_ANE_SHARE_SURFACE_OBJECT=1 restores the old behaviour for a controlled
// reproduction. Without it, h3_ane_run wraps each surface per evaluation.
//
//   xcrun clang -fobjc-arc -framework Foundation -framework Metal \
//     -framework IOSurface -ISources/H3ANEBridge/include \
//     Tools/ANE/pair-stress.m Sources/H3ANEBridge/H3ANEBridge.m -o /tmp/h3-pair-stress
//   /tmp/h3-pair-stress 180        # seconds, default 120
//
// A clean run is evidence, not proof: production shared only the activation and
// took hours to fail, while the keep-alive shared activation and weights and
// took 72 seconds. Time-to-failure is the measurement, so run it longer than
// the failure it is trying to rule out.

#import <Foundation/Foundation.h>
#import <signal.h>
#import "H3ANEBridge.h"

static volatile sig_atomic_t gStop = 0;
static void OnSignal(int sig) { (void)sig; gStop = 1; }

int main(int argc, char **argv) {
    @autoreleasepool {
        signal(SIGINT, OnSignal);
        signal(SIGTERM, OnSignal);
        double limit = argc > 1 ? atof(argv[1]) : 120.0;
        // The variable that actually matters. Continuous submission (gap 0) ran
        // 1.3 million pairs across two 180 s runs without a fault; one pair
        // every 2000 ms panicked the machine in 72 s. The engine powering down
        // between submissions is what distinguishes them, so this sweeps the
        // gap to find where it becomes unsafe.
        double gapMs = argc > 2 ? atof(argv[2]) : 0.0;
        const int d = 64;

        if (!h3_ane_is_available()) { fprintf(stderr, "ANE unavailable.\n"); return 1; }

        H3ANEProgram *p0 = h3_ane_program_create(d, d, d);
        H3ANEProgram *p1 = h3_ane_program_create(d, d, d);
        H3ANETensor *x = h3_ane_tensor_create(d, d);
        H3ANETensor *w = h3_ane_tensor_create(d, d);
        H3ANETensor *y0 = h3_ane_tensor_create(d, d);
        H3ANETensor *y1 = h3_ane_tensor_create(d, d);
        if (!p0 || !p1 || !x || !w || !y0 || !y1) {
            fprintf(stderr, "setup failed\n"); return 2;
        }

        printf("Dual-die pair stress: same x and w bound to both dies, %.0f s, "
               "gap %.0f ms.\n", limit, gapMs);
        printf("surface objects: %s\n\n",
               getenv("H3_ANE_PER_REQUEST_SURFACE_OBJECT")
                   ? "one per evaluation"
                   : "shared across both requests (the default; measured equivalent)");
        fflush(stdout);

        NSDate *start = NSDate.date;
        unsigned long iterations = 0, failures = 0;
        double lastReport = 0;
        while (!gStop) {
            double elapsed = -[start timeIntervalSinceNow];
            if (elapsed >= limit) break;
            if (!h3_ane_run_pair(p0, x, w, y0, p1, x, w, y1)) failures++;
            iterations++;
            if (gapMs > 0) [NSThread sleepForTimeInterval:gapMs / 1000.0];
            if (elapsed - lastReport >= 15.0) {
                lastReport = elapsed;
                printf("  %6.0f s   %8lu pairs   %6.0f pairs/s   %lu failures\n",
                       elapsed, iterations, iterations / elapsed, failures);
                fflush(stdout);
            }
        }

        double total = -[start timeIntervalSinceNow];
        printf("\nSurvived %.1f s, %lu pairs (%.0f/s), %lu submission failures.\n",
               total, iterations, iterations / total, failures);
        printf("The 72 s failure this is testing against is %s.\n",
               total > 72.0 ? "cleared by wall time" : "NOT yet cleared — run longer");
        h3_ane_tensor_free(y1); h3_ane_tensor_free(y0);
        h3_ane_tensor_free(w);  h3_ane_tensor_free(x);
        h3_ane_program_free(p1); h3_ane_program_free(p0);
        return 0;
    }
}

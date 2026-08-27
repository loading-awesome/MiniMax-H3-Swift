// ane-hold.m — hold the ANE's idle timer off for a whole work session.
//
// Run this in its own terminal BEFORE any ANE work, and leave it running until
// the session is over:
//
//   xcrun clang -fobjc-arc -framework Foundation -framework Metal \
//     -framework IOSurface -ISources/H3ANEBridge/include \
//     Tools/ANE/ane-hold.m Sources/H3ANEBridge/H3ANEBridge.m -o /tmp/h3-ane-hold
//   /tmp/h3-ane-hold
//
// WHAT THIS IS FOR
//
// The driver sleeps five seconds after the engine's last work. A program create
// landing inside that 8-15 ms sleep *transition* makes the driver skip the
// action block that completes the transition, and the blocking, gated power-on
// it then issues waits forever inside the IOKit command gate every other ANE
// client needs. The kernel stops making forward progress and the watchdog
// resets the machine: no panic, no log, self-recovery. This happened three
// times on 2026-08-27. See "Machine safety" in docs/ANE_STATUS.md.
//
// Each process that uses the engine and exits leaves the dies idle, and five
// seconds later a transition opens. macOS touches the ANE constantly for its
// own features, so that transition is exposed to accesses this repository does
// not make and cannot see. A per-process keep-alive does nothing about it: it
// dies with the process, right when the window opens. Holding one process for
// the whole session collapses many transitions into one.
//
// WHAT THIS IS NOT
//
// It does not pin the power plane. The supported call for that,
// -[_ANEClient beginRealTimeTask], is entitlement-gated and refuses an unsigned
// process on both connections (measured, sub-millisecond, on retries). This
// tool suppresses the driver's *idle timer* by submitting trivial work more
// often than five seconds; the engine still power-cycles under client-requested
// power-off, and whether that path can race has not been established.
//
// So this narrows the window. It does not close it, and it is not a licence to
// stop being careful about how much engine work gets run.
//
// It deliberately holds no exotic state: it creates one 64x64x64 program
// through the shipping bridge, which starts the bridge's own keep-alive. What
// it holds is therefore exactly what a render holds, not a second mechanism
// that could drift from it.

#import <Foundation/Foundation.h>
#import <signal.h>
#import "H3ANEBridge.h"

static volatile sig_atomic_t gStop = 0;
static void OnSignal(int sig) { (void)sig; gStop = 1; }

int main(int argc, char **argv) {
    @autoreleasepool {
        signal(SIGINT, OnSignal);
        signal(SIGTERM, OnSignal);

        if (!h3_ane_is_available()) {
            fprintf(stderr, "ANE unavailable on this machine or build; nothing to hold.\n");
            return 1;
        }

        // Best effort, and free when it fails. If a signed build ever takes it,
        // this becomes a real power hold rather than a timer suppression.
        bool bracket = h3_ane_power_acquire();

        // The one exposed create of the session. Everything the tests and
        // renders create afterwards happens with the idle timer held off.
        H3ANEProgram *p = h3_ane_program_create(64, 64, 64);
        if (!p || !h3_ane_keepalive_is_running()) {
            fprintf(stderr,
                "The keep-alive is not running, so this tool holds nothing.\n\n"
                "As of 2026-08-27 it is off unless H3_ANE_KEEPALIVE=1, because it is a\n"
                "suspect rather than a fix: 72 seconds of this tool, with nothing else\n"
                "on the machine touching the engine, ended in a kernel panic —\n"
                "  sptm_t8110dart_clear_err: dart-ane0: Unrecoverable secondary error\n"
                "an IOMMU fault on ANE die 0. Do not set that variable to work around\n"
                "this message. See docs/ANE_STATUS.md, 'Machine safety'.\n");
            if (p) h3_ane_program_free(p);
            return 2;
        }

        NSDate *start = NSDate.date;
        printf("ANE hold active.  power bracket: %s (entitlement-gated; expected no)\n",
               bracket ? "held" : "refused");
        printf("Suppressing the driver's 5 s idle timer on both dies. This narrows the\n"
               "race window; it does not pin the power plane.\n\n"
               "Leave this running for the whole session. Verify with:\n"
               "  log show --last 2m --predicate 'eventMessage CONTAINS "
               "\"DriverInitiatedSleepTimerTimeOut\"'\n"
               "Any hit while this runs means the hold is not working.\n\n"
               "Ctrl-C to release.\n\n");
        fflush(stdout);

        unsigned long ticks = 0;
        while (!gStop) {
            [NSThread sleepForTimeInterval:1.0];
            if (++ticks % 60 == 0) {
                printf("  holding %.0f min   keep-alive %s\n",
                       -[start timeIntervalSinceNow] / 60.0,
                       h3_ane_keepalive_is_running() ? "up" : "DOWN");
                fflush(stdout);
            }
        }

        printf("\nReleasing after %.1f min. The dies will idle and the driver will take\n"
               "its normal sleep about five seconds from now — that transition is the\n"
               "exposed one, so do not start ANE work again without restarting this.\n",
               -[start timeIntervalSinceNow] / 60.0);
        h3_ane_program_free(p);
        h3_ane_power_release();
        return 0;
    }
}

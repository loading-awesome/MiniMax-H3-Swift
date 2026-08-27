// power-bracket-check.m — does the ANE power bracket work, and does it hold?
//
// The smallest thing that touches the engine. It creates no program, loads no
// model, allocates no IOSurface, and never evaluates — so it cannot take the
// create-versus-sleep race that hard-locked this machine on 2026-08-27. All it
// does is call the bracket that every CoreML client calls constantly, then sit
// still long enough for the driver's five-second idle timer to fire twice.
//
// It calls the shipping bridge rather than a copy of it, so what passes here is
// what production runs.
//
// Two questions, both unanswered before this runs:
//
//   1. Does `beginRealTimeTask` succeed for an unentitled process over
//      `+[_ANEClient sharedConnection]`, or does it need the restricted
//      connection and `com.apple.aned.private.allow`? If it fails, the bridge
//      now refuses to create programs at all and `H3_ANE=experimental` becomes
//      a silent GPU fallback.
//   2. Does holding it actually suppress the driver's sleep? Answer that from
//      the log, not from this program's output:
//
//        log show --last 2m --predicate 'eventMessage CONTAINS "DriverInitiated"'
//
//      A `DriverInitiatedSleepTimerTimeOut` inside the idle window means the
//      bracket does not hold power on its own and a sub-five-second keep-alive
//      is also required.
//
//   xcrun clang -fobjc-arc -framework Foundation -framework Metal \
//     -framework IOSurface -ISources/H3ANEBridge/include \
//     Tools/ANE/power-bracket-check.m Sources/H3ANEBridge/H3ANEBridge.m \
//     -o /tmp/h3-power-bracket-check

#import <Foundation/Foundation.h>
#import "H3ANEBridge.h"

int main(int argc, char **argv) {
    @autoreleasepool {
        int idle = argc > 1 ? atoi(argv[1]) : 30;

        printf("h3_ane_is_available     = %s\n", h3_ane_is_available() ? "yes" : "NO");
        if (!h3_ane_is_available()) {
            printf("\nThe bridge refuses this machine or this runtime. With the bracket now\n"
                   "part of the validated ABI, that also means the ANE selectors it needs\n"
                   "are missing. Nothing further to test.\n");
            return 1;
        }

        NSDate *t0 = NSDate.date;
        bool ok = h3_ane_power_acquire();
        double ms = -[t0 timeIntervalSinceNow] * 1000;
        printf("h3_ane_power_acquire    = %s  (%.1f ms)\n", ok ? "yes" : "NO", ms);
        printf("h3_ane_power_is_held    = %s\n", h3_ane_power_is_held() ? "yes" : "no");

        if (!ok) {
            printf("\nExpected: the bracket is entitlement-gated and refuses this\n"
                   "process on both connections. The keep-alive is what carries the\n"
                   "safety now, so that is what this tests.\n");
        }

        // The one exposed operation: a create, which starts the keep-alive
        // before it creates anything of its own.
        NSDate *t1 = NSDate.date;
        H3ANEProgram *p = h3_ane_program_create(64, 64, 64);
        printf("h3_ane_program_create   = %s  (%.1f ms)\n", p ? "ok" : "NULL",
               -[t1 timeIntervalSinceNow] * 1000);
        printf("keepalive_is_running    = %s\n",
               h3_ane_keepalive_is_running() ? "yes" : "NO");
        if (!p || !h3_ane_keepalive_is_running()) {
            printf("\nThe keep-alive did not come up, so the bridge is refusing to create\n"
                   "programs. That is the intended failure: a slower render, not a lost\n"
                   "machine.\n");
            return 2;
        }

        printf("\nIdling %d s. The driver's idle timer is 5 s, so without the keep-alive\n"
               "this window would contain several sleep timeouts. The log is the verdict,\n"
               "not this program's output.\n", idle);
        fflush(stdout);
        [NSThread sleepForTimeInterval:idle];

        printf("keepalive still running = %s\n",
               h3_ane_keepalive_is_running() ? "yes" : "NO");
        h3_ane_program_free(p);
        printf("\nNow read the log for the idle window:\n"
               "  log show --last %dm --predicate 'eventMessage CONTAINS \"DriverInitiatedSleepTimerTimeOut\"'\n"
               "Any hit inside the idle window means the keep-alive is not holding and the\n"
               "interval needs to come down.\n", idle / 60 + 1);
        return 0;
    }
}

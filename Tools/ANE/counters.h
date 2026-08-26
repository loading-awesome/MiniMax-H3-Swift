// counters.h — Whole-engine ANE telemetry through IOReport, per die.
//
// The engine's twenty-four per-task-descriptor performance counters are gated:
// a kernel check rejects the load when the stats mask is non-zero on the
// unentitled path, because the compiled program carries no stats-descriptor
// section for the kernel to size. What sits outside that gate is whole-engine
// telemetry, and it needs no entitlement — only the same runtime symbol lookup
// the rest of Tools/ANE already does.
//
// WHAT THIS HOST ACTUALLY EXPOSES, enumerated rather than assumed. The
// published DRAM-byte channels (`AMC Stats|Perf Counters|ANE0 RD`) are from M1
// and M5 hosts and DO NOT EXIST on this one: there is no `AMC Stats` group at
// all, and the per-agent `PMP*|AGENT BW` group lists only IO agents (IOA5,
// IOA6, IOA13, IOA14), no engine. So engine DRAM traffic is not measurable
// here. What replaces it is `PMP*|DCS BW`, a 32-bin bandwidth histogram per
// memory controller (64 GB/s per bin, to 2048 GB/s) reporting TOTAL DRAM
// bandwidth rather than the engine's share — which suits the question better
// than the counter that went missing. "Is there spare bandwidth" is answered
// by total demand against the ~800 GB/s ceiling, and measured that way there
// is no contention to find: peak read stays at or below 192 GB/s across every
// arm. Four channel groups are real:
//
//   Energy Model | ANE0_0 / ANE0_1            per-die engine energy  <- the one that works
//   SoC Stats | Cluster Power States | *ANE*  per-die DVFS residency
//   SoC Stats | Events | *_ANE_*TRG           throttle triggers
//   PMP0/1 | DCS BW | AMCC0/1 RD, WR           total DRAM bandwidth, per die
//
// Per-die energy is what makes this worth having. The wall clock cannot say
// WHICH die ran a job — `ane2_isolated_ms` reads the same whichever engine
// executed it — and energy can: a die that did nothing reports exactly 0 mJ.
// That is how the instance-hint result was found. The throttle counters are a
// validity check, since an overlap comparison taken across a clock event is
// not a comparison.
//
// Matching is on exact group and subgroup, never a substring of the channel
// name. "ANE" is a substring of "Miscellaneous", "VLane" and "LanesEng", and a
// loose filter silently pulls in network and storage counters.

#pragma once
#include <CoreFoundation/CoreFoundation.h>
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

typedef CFDictionaryRef IOReportSampleRef;
enum { kANEFormatSimple = 1, kANEFormatState = 2 };

typedef CFMutableDictionaryRef (*ANEIRCopyAllFn)(uint64_t, uint64_t);
typedef void *(*ANEIRSubscribeFn)(void *, CFMutableDictionaryRef, CFMutableDictionaryRef *,
                                  uint64_t, CFTypeRef);
typedef CFDictionaryRef (*ANEIRSamplesFn)(void *, CFMutableDictionaryRef, CFTypeRef);
typedef CFDictionaryRef (*ANEIRDeltaFn)(CFDictionaryRef, CFDictionaryRef, CFTypeRef);
typedef void (*ANEIRIterateFn)(CFDictionaryRef, int (^)(IOReportSampleRef));
typedef int64_t (*ANEIRIntFn)(CFDictionaryRef, int);
typedef CFStringRef (*ANEIRStrFn)(CFDictionaryRef);
typedef int32_t (*ANEIRFormatFn)(CFDictionaryRef);
typedef int32_t (*ANEIRStateCountFn)(CFDictionaryRef);
typedef CFStringRef (*ANEIRStateNameFn)(CFDictionaryRef, int32_t);
typedef int64_t (*ANEIRResidencyFn)(CFDictionaryRef, int32_t);

typedef struct {
    int available;
    void *subscription;
    CFMutableDictionaryRef subscribed;
    CFDictionaryRef opening;
    ANEIRSamplesFn    samples;
    ANEIRDeltaFn      delta;
    ANEIRIterateFn    iterate;
    ANEIRIntFn        integerValue;
    ANEIRStrFn        group, subGroup, channelName;
    ANEIRFormatFn     format;
    ANEIRStateCountFn stateCount;
    ANEIRStateNameFn  stateName;
    ANEIRResidencyFn  residency;
} ANECounters;

typedef struct {
    double energyMillijoules[2];   // per die, as reported by the Energy Model
    double activeTicks[2];         // per die, DVFS "ACT" residency (clock-up, not work)
    double idleTicks[2];           // per die, "INACT" residency
    long   throttleEvents;         // ADCLK / dither / peak-power triggers
    double readGBs[2], writeGBs[2];   // per die, residency-weighted DRAM bandwidth
    double peakReadGBs[2], peakWriteGBs[2];  // per die, highest bin entered
    double aboveFloorPct[2];          // per die, % of read residency above bin 0
    int    energyChannels, residencyChannels, bandwidthChannels;
} ANECounterDelta;

static int ANECFEquals(CFStringRef s, const char *literal) {
    if (!s) return 0;
    char buffer[160];
    if (!CFStringGetCString(s, buffer, sizeof buffer, kCFStringEncodingUTF8)) return 0;
    return strcmp(buffer, literal) == 0;
}

static int ANECFCopy(CFStringRef s, char *out, size_t n) {
    out[0] = 0;
    return s && CFStringGetCString(s, out, (CFIndex)n, kCFStringEncodingUTF8);
}

// ANE0_0 is die 0 and ANE0_1 is die 1 on a two-die part; DIE_0_ANE0 and
// DIE_1_ANE0 name the same two clusters in the power-state group.
static int ANEDieOf(const char *name) {
    if (strstr(name, "DIE_1") || strstr(name, "ANE0_1") || strstr(name, "ANE1")) return 1;
    return 0;
}

static ANECounters ANECountersOpen(void) {
    ANECounters c; memset(&c, 0, sizeof c);
    void *lib = dlopen("/usr/lib/libIOReport.dylib", RTLD_NOW);
    if (!lib) return c;
    ANEIRCopyAllFn   copyAll   = (ANEIRCopyAllFn)dlsym(lib, "IOReportCopyAllChannels");
    ANEIRSubscribeFn subscribe = (ANEIRSubscribeFn)dlsym(lib, "IOReportCreateSubscription");
    c.samples      = (ANEIRSamplesFn)dlsym(lib, "IOReportCreateSamples");
    c.delta        = (ANEIRDeltaFn)dlsym(lib, "IOReportCreateSamplesDelta");
    c.iterate      = (ANEIRIterateFn)dlsym(lib, "IOReportIterate");
    c.integerValue = (ANEIRIntFn)dlsym(lib, "IOReportSimpleGetIntegerValue");
    c.group        = (ANEIRStrFn)dlsym(lib, "IOReportChannelGetGroup");
    c.subGroup     = (ANEIRStrFn)dlsym(lib, "IOReportChannelGetSubGroup");
    c.channelName  = (ANEIRStrFn)dlsym(lib, "IOReportChannelGetChannelName");
    c.format       = (ANEIRFormatFn)dlsym(lib, "IOReportChannelGetFormat");
    c.stateCount   = (ANEIRStateCountFn)dlsym(lib, "IOReportStateGetCount");
    c.stateName    = (ANEIRStateNameFn)dlsym(lib, "IOReportStateGetNameForIndex");
    c.residency    = (ANEIRResidencyFn)dlsym(lib, "IOReportStateGetResidency");
    if (!copyAll || !subscribe || !c.samples || !c.delta || !c.iterate ||
        !c.integerValue || !c.channelName || !c.format) return c;

    CFMutableDictionaryRef all = copyAll(0, 0);
    if (!all) return c;
    c.subscription = subscribe(NULL, all, &c.subscribed, 0, NULL);
    CFRelease(all);
    if (!c.subscription || !c.subscribed) return c;
    c.available = 1;
    return c;
}

static void ANECountersBegin(ANECounters *c) {
    if (!c->available) return;
    if (c->opening) { CFRelease(c->opening); c->opening = NULL; }
    c->opening = c->samples(c->subscription, c->subscribed, NULL);
}

static ANECounterDelta ANECountersEnd(ANECounters *c) {
    __block ANECounterDelta d; memset(&d, 0, sizeof d);
    if (!c->available || !c->opening) return d;
    CFDictionaryRef now = c->samples(c->subscription, c->subscribed, NULL);
    if (!now) return d;
    CFDictionaryRef delta = c->delta(c->opening, now, NULL);
    ANECounters *self = c;
    if (delta) {
        c->iterate(delta, ^int(IOReportSampleRef ch) {
            char g[160], s[160], n[160];
            ANECFCopy(self->group(ch), g, sizeof g);
            ANECFCopy(self->subGroup ? self->subGroup(ch) : NULL, s, sizeof s);
            ANECFCopy(self->channelName(ch), n, sizeof n);

            if (!strcmp(g, "Energy Model") && !strncmp(n, "ANE", 3)) {
                d.energyMillijoules[ANEDieOf(n)] += (double)self->integerValue(ch, 0);
                ++d.energyChannels;
                return 0;
            }
            if (!strcmp(s, "Cluster Power States") && strstr(n, "ANE")) {
                if (self->format(ch) != kANEFormatState || !self->stateCount) return 0;
                int die = ANEDieOf(n);
                int32_t count = self->stateCount(ch);
                for (int32_t i = 0; i < count; ++i) {
                    char state[64];
                    ANECFCopy(self->stateName(ch, i), state, sizeof state);
                    double ticks = (double)self->residency(ch, i);
                    if (ticks <= 0) continue;
                    // This host names the two states "ACT" and "INACT". Match
                    // exactly: "ACT" is a substring of "INACT", so a contains
                    // test files every idle tick as work and the counter reads
                    // busy on a machine doing nothing.
                    if (!strcmp(state, "ACT")) d.activeTicks[die] += ticks;
                    else                       d.idleTicks[die]   += ticks;
                }
                ++d.residencyChannels;
                return 0;
            }
            // 32 bins labelled by their upper edge ("  64GB/s"). The weighted
            // mean is therefore an upper bound on true bandwidth, which is
            // fine for comparing arms measured the same way. The peak bin
            // entered is reported alongside, because a contention story shows
            // up as a ceiling on the peak rather than a shift in the mean.
            if (!strcmp(s, "DCS BW") && strstr(n, "AMCC") &&
                (strstr(n, " RD") || strstr(n, " WR")) && !strstr(n, "RD+WR")) {
                if (self->format(ch) != kANEFormatState || !self->stateCount) return 0;
                int die = strstr(n, "AMCC1") ? 1 : 0;
                int isRead = strstr(n, " RD") != NULL;
                double weighted = 0, total = 0, peak = 0, floorTicks = 0;
                int32_t count = self->stateCount(ch);
                for (int32_t i = 0; i < count; ++i) {
                    char bin[64];
                    ANECFCopy(self->stateName(ch, i), bin, sizeof bin);
                    double edge = atof(bin);          // leading spaces are fine
                    double ticks = (double)self->residency(ch, i);
                    if (ticks <= 0) continue;
                    weighted += edge * ticks;
                    total += ticks;
                    if (i == 0) floorTicks = ticks;
                    if (edge > peak) peak = edge;
                }
                if (total > 0) {
                    if (isRead) {
                        d.readGBs[die] = weighted / total;
                        d.peakReadGBs[die] = peak;
                        d.aboveFloorPct[die] = 100.0 * (total - floorTicks) / total;
                    } else {
                        d.writeGBs[die] = weighted / total;
                        d.peakWriteGBs[die] = peak;
                    }
                    ++d.bandwidthChannels;
                }
                return 0;
            }
            if (!strcmp(s, "Events") && strstr(n, "_ANE_") && strstr(n, "TRG")) {
                int64_t v = self->integerValue(ch, 0);
                if (v > 0) d.throttleEvents += (long)v;
                return 0;
            }
            return 0;
        });
        CFRelease(delta);
    }
    CFRelease(now);
    return d;
}

static void ANECountersReport(const char *label, ANECounterDelta d) {
    if (!d.energyChannels && !d.residencyChannels) {
        printf("  %-22s no ANE telemetry channels matched\n", label);
        return;
    }
    // Energy is the work signal: simple-format, reads exactly 0 on a die that
    // did nothing, and repeats to under 1% across runs.
    //
    // Clock-up is DVFS residency, which is NOT the same question. The engine
    // stays in ACT for some idle timeout after its last dispatch, so an arm
    // that ran no ANE work still shows clock-up inherited from the arm before
    // it. Read it one way only: 0% is conclusive that the die did nothing,
    // and anything above 0% is not evidence that it did. Its tick base is also
    // not wall time — ACT+INACT does not scale linearly with the interval — so
    // only the ratio is reported, never the raw counts.
    double clock0 = d.activeTicks[0] + d.idleTicks[0];
    double clock1 = d.activeTicks[1] + d.idleTicks[1];
    printf("  %-22s energy d0=%7.1f mJ  d1=%7.1f mJ | clock-up d0=%3.0f%% d1=%3.0f%%",
           label, d.energyMillijoules[0], d.energyMillijoules[1],
           clock0 > 0 ? 100.0 * d.activeTicks[0] / clock0 : 0.0,
           clock1 > 0 ? 100.0 * d.activeTicks[1] / clock1 : 0.0);
    if (d.throttleEvents) printf("  THROTTLED x%ld", d.throttleEvents);
    printf("\n");
    // Bin 0's label is 64 GB/s and it covers everything below that, so the
    // weighted mean can never read under 64 and is an upper bound, not a
    // measurement. Peak bin entered and time above the floor are the honest
    // signals: they say how close the machine came to its ~800 GB/s ceiling.
    if (d.bandwidthChannels)
        printf("  %-22s DRAM peak rd/wr  d0 %4.0f/%4.0f  d1 %4.0f/%4.0f GB/s   "
               "above 64 GB/s: d0 %4.1f%% d1 %4.1f%%\n", "",
               d.peakReadGBs[0], d.peakWriteGBs[0],
               d.peakReadGBs[1], d.peakWriteGBs[1],
               d.aboveFloorPct[0], d.aboveFloorPct[1]);
}

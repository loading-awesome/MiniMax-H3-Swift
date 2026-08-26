// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

#import "H3ANEBridge.h"
#import <Foundation/Foundation.h>
#import <math.h>

int main(void) {
    @autoreleasepool {
        if (!h3_ane_is_available()) {
            fprintf(stderr, "H3ANEBridge is unavailable on this machine/build\n");
            return 77;
        }

        const int s = 64, k = 128, n = 32;
        H3ANEProgram *program = h3_ane_program_create(s, k, n);
        H3ANETensor *x = h3_ane_tensor_create(k, s);
        H3ANETensor *w = h3_ane_tensor_create(k, n);
        H3ANETensor *y = h3_ane_tensor_create(s, n);
        if (!program || !x || !w || !y) {
            fprintf(stderr, "ANE program/tensor setup failed\n");
            h3_ane_program_free(program);
            h3_ane_tensor_free(x); h3_ane_tensor_free(w); h3_ane_tensor_free(y);
            return 1;
        }

        _Float16 *xData = calloc((size_t)k * s, sizeof(_Float16));
        _Float16 *wData = calloc((size_t)k * n, sizeof(_Float16));
        for (int i = 0; i < k * s; ++i) xData[i] = (_Float16)0.0625f;
        for (int i = 0; i < k * n; ++i) wData[i] = (_Float16)1.0f;

        bool ok = xData && wData &&
            h3_ane_tensor_write(x, xData, k, s) &&
            h3_ane_tensor_write(w, wData, k, n) &&
            h3_ane_run(program, x, w, y, 1);
        _Float16 *result = ok ? (_Float16 *)h3_ane_tensor_ptr(y) : NULL;
        float first = result ? (float)result[0] : NAN;
        ok = ok && isfinite(first) && fabsf(first - 8.0f) < 0.05f;

        printf("H3ANEBridge standalone: %s (y[0]=%.4f, expected 8.0)\n",
               ok ? "PASS" : "FAIL", first);

        free(xData); free(wData);
        h3_ane_program_free(program);
        h3_ane_tensor_free(x); h3_ane_tensor_free(w); h3_ane_tensor_free(y);
        return ok ? 0 : 2;
    }
}

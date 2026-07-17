/*
 *
 * Standard interface for C programs compiled to Transformer VM via WebAssembly.
 *
 * Entry point: void compute(const unsigned char *input)
 *   - `input` points to the payload in a length-delimited linear-memory input
 *   - `tvm_input_length(input)` returns the payload length in bytes
 *   - a compatibility NUL follows the payload for legacy string programs
 *   - Use putchar() to emit output bytes
 *
 * All helpers use always_inline and avoid arrays / address-taken locals
 * so that clang can place everything in WASM registers (no stack frame).
 *
 * Compile with:
 *   clang --target=wasm32 -nostdlib -Oz \
 *     -Wl,--no-entry -Wl,--export=compute \
 *     -Wl,--initial-memory=131072 \
 *     -o out.wasm source.c
 */

#ifndef TRANSFORMER_VM_RUNTIME_H
#define TRANSFORMER_VM_RUNTIME_H

/* ── Output ─────────────────────────────────────────────────────── */

__attribute__((import_module("env"), import_name("output_byte")))
void putchar(int ch);

/* Input bytes are preceded by their unsigned little-endian 32-bit length. */
__attribute__((always_inline))
static inline unsigned int tvm_input_length(const unsigned char *input) {
    const unsigned char *length = input - 4;
    return (unsigned int)length[0] | ((unsigned int)length[1] << 8) |
           ((unsigned int)length[2] << 16) | ((unsigned int)length[3] << 24);
}

/* ── Helpers ────────────────────────────────────────────────────── */

__attribute__((always_inline))
static inline unsigned int tvm_addcarry_u32(unsigned int a, unsigned int b,
                                             unsigned int carry_in,
                                             unsigned int *carry_out) {
    unsigned int sum = a + b;
    unsigned int carry = sum < a;
    unsigned int carry_bit = carry_in != 0u;
    unsigned int result = sum + carry_bit;
    *carry_out = carry | (result < sum);
    return result;
}

__attribute__((always_inline))
static inline unsigned int tvm_subborrow_u32(unsigned int a, unsigned int b,
                                              unsigned int borrow_in,
                                              unsigned int *borrow_out) {
    unsigned int difference = a - b;
    unsigned int borrow = a < b;
    unsigned int borrow_bit = borrow_in != 0u;
    unsigned int result = difference - borrow_bit;
    *borrow_out = borrow | (difference < borrow_bit);
    return result;
}

__attribute__((always_inline))
static inline unsigned int tvm_load32_le(const unsigned char *p) {
    return (unsigned int)p[0] | ((unsigned int)p[1] << 8) |
           ((unsigned int)p[2] << 16) | ((unsigned int)p[3] << 24);
}

__attribute__((always_inline))
static inline unsigned int tvm_load32_be(const unsigned char *p) {
    return ((unsigned int)p[0] << 24) | ((unsigned int)p[1] << 16) |
           ((unsigned int)p[2] << 8) | (unsigned int)p[3];
}

__attribute__((always_inline))
static inline void tvm_store32_le(unsigned char *p, unsigned int value) {
    p[0] = (unsigned char)value;
    p[1] = (unsigned char)(value >> 8);
    p[2] = (unsigned char)(value >> 16);
    p[3] = (unsigned char)(value >> 24);
}

__attribute__((always_inline))
static inline void tvm_store32_be(unsigned char *p, unsigned int value) {
    p[0] = (unsigned char)(value >> 24);
    p[1] = (unsigned char)(value >> 16);
    p[2] = (unsigned char)(value >> 8);
    p[3] = (unsigned char)value;
}

/* Branchless word selection. This does not make VM execution constant-time. */
__attribute__((always_inline))
static inline unsigned int tvm_select_u32(unsigned int when_true,
                                           unsigned int when_false,
                                           unsigned int condition) {
    unsigned int mask = 0u - (condition != 0u);
    return (when_true & mask) | (when_false & ~mask);
}

__attribute__((always_inline))
static inline int str_len(const char *s) {
    int n = 0;
    while (s[n]) n++;
    return n;
}

__attribute__((always_inline))
static inline void print_str(const char *s) {
    while (*s) putchar(*s++);
}

/* Parse a non-negative decimal integer.
 * Uses repeated addition (not MUL). No pointer-to-pointer needed. */
__attribute__((always_inline))
static inline int parse_int(const char *p) {
    int n = 0;
    while (*p >= '0' && *p <= '9') {
        int d = *p - '0';
        int t2 = n + n;
        int t4 = t2 + t2;
        int t8 = t4 + t4;
        n = t8 + t2 + d;
        p = p + 1;
    }
    return n;
}

/* Print a non-negative integer without using any arrays.
 * Finds the largest power-of-10 <= n, then extracts digits
 * from most significant to least via subtraction. */
__attribute__((always_inline))
static inline void print_int(int n) {
    if (n < 0) { putchar('-'); n = 0 - n; }
    if (n == 0) { putchar('0'); return; }
    /* Find largest power of 10 <= n */
    int p = 1;
    int tmp = n;
    while (tmp >= 10) {
        int q = 0, r = tmp;
        while (r >= 10) { r = r - 10; q = q + 1; }
        int p2 = p + p;
        int p4 = p2 + p2;
        int p8 = p4 + p4;
        p = p8 + p2;
        tmp = q;
    }
    /* Print digits from MSD to LSD */
    while (p > 0) {
        int d = 0;
        while (n >= p) { n = n - p; d = d + 1; }
        putchar('0' + d);
        int q = 0, r = p;
        while (r >= 10) { r = r - 10; q = q + 1; }
        p = q;
    }
}

/* Minimal sscanf supporting %d only.
 * Parses integers from str according to fmt, storing into int* args.
 * Skips leading whitespace before %d. Returns number of items matched. */
__attribute__((noinline))
static int sscanf(const char *str, const char *fmt, ...) {
    __builtin_va_list ap;
    __builtin_va_start(ap, fmt);
    int matched = 0;
    while (*fmt && *str) {
        if (*fmt == '%') {
            fmt++;
            if (*fmt == 'd') {
                /* skip whitespace */
                while (*str == ' ' || *str == '\t' || *str == '\n') str++;
                if (!*str) break;
                int neg = 0;
                if (*str == '-') { neg = 1; str++; }
                if (*str < '0' || *str > '9') break;
                int n = 0;
                while (*str >= '0' && *str <= '9') {
                    int d = *str - '0';
                    int t2 = n + n;
                    int t4 = t2 + t2;
                    int t8 = t4 + t4;
                    n = t8 + t2 + d;
                    str++;
                }
                if (neg) n = 0 - n;
                *__builtin_va_arg(ap, int*) = n;
                matched++;
                fmt++;
            } else {
                fmt++;
            }
        } else if (*fmt == ' ') {
            /* whitespace in fmt matches zero or more whitespace in str */
            while (*str == ' ' || *str == '\t' || *str == '\n') str++;
            fmt++;
        } else {
            if (*str != *fmt) break;
            str++;
            fmt++;
        }
    }
    __builtin_va_end(ap);
    return matched;
}

/* Basic printf supporting %d, %s, %c, and %%. */
__attribute__((noinline))
static void printf(const char *fmt, ...) {
    __builtin_va_list ap;
    __builtin_va_start(ap, fmt);
    while (*fmt) {
        if (*fmt == '%') {
            fmt++;
            if (*fmt == 'd')      print_int(__builtin_va_arg(ap, int));
            else if (*fmt == 's') print_str(__builtin_va_arg(ap, const char *));
            else if (*fmt == 'c') putchar(__builtin_va_arg(ap, int));
            else if (*fmt == '%') putchar('%');
            if (*fmt) fmt++;
        } else {
            putchar(*fmt++);
        }
    }
    __builtin_va_end(ap);
}

#endif /* TRANSFORMER_VM_RUNTIME_H */

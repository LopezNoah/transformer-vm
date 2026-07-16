/* Differential conformance fixture for arithmetic, control flow, memory,
 * locals, and calls. Input is a non-negative decimal integer. */

__attribute__((noinline))
static int helper(int value) {
    return value + 3;
}

void compute(const char *input) {
    unsigned value = (unsigned)parse_int(input);
    int local = helper((int)value);
    int total = 0;
    int i = 0;
    while (i < 4) {
        total = total + i;
        i = i + 1;
    }

    int *cell = (int *)128;
    *cell = local + total;
    if ((value & 1) != 0) {
        *cell = *cell + 7;
    } else {
        *cell = *cell - 2;
    }

    /* Verify signed comparison at the 32-bit boundary and unsigned wrap. */
    unsigned boundary = 0x7fffffffu;
    boundary = boundary + 1;
    if ((int)boundary < 0) {
        *cell = *cell + 1;
    }
    boundary = 0xffffffffu;
    boundary = boundary + 1;
    if (boundary == 0) {
        *cell = *cell + 1;
    }

    putchar(*cell & 0x7f);
}

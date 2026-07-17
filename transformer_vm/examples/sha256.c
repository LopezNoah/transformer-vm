/*
 * Portable SHA-256 example for Transformer VM.
 *
 * Input is the binary-safe Transformer VM payload. Output is a lowercase
 * hexadecimal SHA-256 digest.
 */

static const unsigned int round_constants[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
    0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
    0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
    0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
    0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u
};

static unsigned int rotate_right(unsigned int value, unsigned int count) {
    return (value >> count) | (value << (32u - count));
}

static unsigned int load_be32(const unsigned char *p) {
    return ((unsigned int)p[0] << 24) | ((unsigned int)p[1] << 16) |
           ((unsigned int)p[2] << 8) | (unsigned int)p[3];
}

static void compress(unsigned int state[8], const unsigned char block[64]) {
    unsigned int words[64];
    unsigned int a, b, c, d, e, f, g, h;
    unsigned int i;

    for (i = 0; i < 16; i = i + 1) words[i] = load_be32(block + i * 4u);
    for (i = 16; i < 64; i = i + 1) {
        unsigned int x = words[i - 15];
        unsigned int y = words[i - 2];
        unsigned int s0 = rotate_right(x, 7) ^ rotate_right(x, 18) ^ (x >> 3);
        unsigned int s1 = rotate_right(y, 17) ^ rotate_right(y, 19) ^ (y >> 10);
        words[i] = words[i - 16] + s0 + words[i - 7] + s1;
    }

    a = state[0]; b = state[1]; c = state[2]; d = state[3];
    e = state[4]; f = state[5]; g = state[6]; h = state[7];
    for (i = 0; i < 64; i = i + 1) {
        unsigned int sum1 = rotate_right(e, 6) ^ rotate_right(e, 11) ^ rotate_right(e, 25);
        unsigned int choice = (e & f) ^ ((~e) & g);
        unsigned int temp1 = h + sum1 + choice + round_constants[i] + words[i];
        unsigned int sum0 = rotate_right(a, 2) ^ rotate_right(a, 13) ^ rotate_right(a, 22);
        unsigned int majority = (a & b) ^ (a & c) ^ (b & c);
        unsigned int temp2 = sum0 + majority;

        h = g; g = f; f = e; e = d + temp1;
        d = c; c = b; b = a; a = temp1 + temp2;
    }

    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

static void print_hex_byte(unsigned int value) {
    static const char digits[] = "0123456789abcdef";
    putchar(digits[(value >> 4) & 15u]);
    putchar(digits[value & 15u]);
}

void compute(const char *input) {
    const unsigned char *message = (const unsigned char *)input;
    unsigned int length = tvm_input_length(message);
    unsigned int state[8] = {
        0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u
    };
    unsigned char final_blocks[128];
    unsigned int offset = 0;
    unsigned int remainder, final_length, bit_length_high, bit_length_low;
    unsigned int i;

    while (length - offset >= 64u) {
        compress(state, message + offset);
        offset += 64u;
    }

    remainder = length - offset;
    final_length = remainder < 56u ? 64u : 128u;
    for (i = 0; i < remainder; i = i + 1) final_blocks[i] = message[offset + i];
    final_blocks[remainder] = 0x80u;
    for (i = remainder + 1; i < final_length; i = i + 1) final_blocks[i] = 0;

    bit_length_high = length >> 29;
    bit_length_low = length << 3;
    final_blocks[final_length - 8] = bit_length_high >> 24;
    final_blocks[final_length - 7] = bit_length_high >> 16;
    final_blocks[final_length - 6] = bit_length_high >> 8;
    final_blocks[final_length - 5] = bit_length_high;
    final_blocks[final_length - 4] = bit_length_low >> 24;
    final_blocks[final_length - 3] = bit_length_low >> 16;
    final_blocks[final_length - 2] = bit_length_low >> 8;
    final_blocks[final_length - 1] = bit_length_low;

    compress(state, final_blocks);
    if (final_length == 128u) compress(state, final_blocks + 64);

    for (i = 0; i < 8; i = i + 1) {
        print_hex_byte(state[i] >> 24);
        print_hex_byte(state[i] >> 16);
        print_hex_byte(state[i] >> 8);
        print_hex_byte(state[i]);
    }
    putchar('\n');
}

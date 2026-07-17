# i32 Conformance Matrix

This matrix covers the supported MVP `i32` operations. Results are represented as
unsigned 32-bit words; signed operations use two's-complement interpretation.

| Operation group | Operations | Semantics | Trap behavior |
| --- | --- | --- | --- |
| Constants | `i32.const` | Low 32 bits of the immediate | Never |
| Arithmetic | `i32.add`, `i32.sub` | Modulo 2^32 | Never |
| Division | `i32.div_u`, `i32.div_s` | Unsigned or truncation-toward-zero signed quotient | Zero divisor; `INT_MIN / -1` for signed division |
| Remainder | `i32.rem_u`, `i32.rem_s` | Unsigned remainder; signed remainder has dividend sign | Zero divisor |
| Bitwise | `i32.and`, `i32.or`, `i32.xor` | Bitwise operation | Never |
| Shifts and rotates | `i32.shl`, `i32.shr_u`, `i32.shr_s`, `i32.rotl`, `i32.rotr` | Shift count masked to five bits | Never |
| Comparisons | `i32.eqz`, `i32.eq`, `i32.ne`, `i32.lt_*`, `i32.gt_*`, `i32.le_*`, `i32.ge_*` | Returns `0` or `1`; `_s` is signed and `_u` unsigned | Never |
| Sign extension | `i32.extend8_s`, `i32.extend16_s` | Sign-extend low 8 or 16 bits | Never |
| Loads | `i32.load*` | Little-endian, including signed narrow loads | Effective address plus access width outside memory |
| Stores | `i32.store*` | Little-endian low 8, 16, or 32 bits | Effective address plus access width outside memory |

The compiler lowers division, remainder, sign extension, and bounds checks to the
base interpreter instruction set. The reference executor applies the same traps
before touching linear memory.

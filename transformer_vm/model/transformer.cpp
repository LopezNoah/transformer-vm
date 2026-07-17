/*
 * transformer.cpp — Standalone C++ inference for the WASM transformer.
 *
 * Loads weights from a binary file (build_model.py --save-weights=...)
 * and runs autoregressive generation with O(log n) hard attention.
 *
 * Build:  make transformer
 * Run:    ./transformer model.bin data/collatz.txt
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef __APPLE__
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#endif

#include "hull2d_cht.h"

static constexpr int MAX_GEN = 6000000;

// ── Model ──────────────────────────────────────────────────────────────

struct SparseMatrix {
    std::vector<double> val;
    std::vector<int>    col;
    std::vector<int>    ptr;  // ptr[i] = start of row i in val/col
    int rows = 0, cols = 0, nnz = 0;

    void build(const double* dense, int r, int c) {
        rows = r; cols = c; nnz = 0;
        val.clear(); col.clear();
        ptr.resize(r + 1);
        for (int i = 0; i < r; i++) {
            ptr[i] = nnz;
            for (int j = 0; j < c; j++) {
                if (dense[i * c + j] != 0.0) {
                    val.push_back(dense[i * c + j]);
                    col.push_back(j);
                    nnz++;
                }
            }
        }
        ptr[r] = nnz;
    }
};

struct DenseMatrix {
    std::vector<double> val;
    int rows = 0, cols = 0;

    void build(const SparseMatrix& sparse) {
        rows = sparse.rows; cols = sparse.cols;
        val.assign((size_t)rows * cols, 0.0);
        for (int i = 0; i < rows; i++)
            for (int k = sparse.ptr[i]; k < sparse.ptr[i + 1]; k++)
                val[(size_t)i * cols + sparse.col[k]] = sparse.val[k];
    }
};

struct Layer {
    SparseMatrix qkv, out, fi, fo;
    DenseMatrix dense_qkv, dense_out, dense_fi, dense_fo;
    std::vector<bool> active_heads;
    bool active = true;
};

struct Model {
    int V, D, L, H, F, stop;
    std::vector<std::string> name;
    std::unordered_map<std::string, int> id;
    SparseMatrix emb;
    DenseMatrix dense_emb;
    std::vector<Layer> ly;
    SparseMatrix head;
    DenseMatrix dense_head;
    bool dense = false;
    std::vector<std::vector<int>> attn_erase;
    std::vector<std::vector<int>> ffn_erase;
    std::vector<std::vector<TieBreak>> head_tb;
    std::vector<std::vector<int>> head_type;
};

static void read_sparse(FILE* f, SparseMatrix& s) {
    uint32_t rows, cols;
    uint64_t nnz;
    if (fread(&rows, 4, 1, f) != 1 || fread(&cols, 4, 1, f) != 1 || fread(&nnz, 8, 1, f) != 1)
        { fprintf(stderr, "truncated sparse matrix header\n"); exit(1); }
    s.rows = rows; s.cols = cols; s.nnz = nnz;
    s.ptr.resize((size_t)rows + 1); s.col.resize(nnz); s.val.resize(nnz);
    if (fread(s.ptr.data(), 4, (size_t)rows + 1, f) != (size_t)rows + 1
        || fread(s.col.data(), 4, nnz, f) != nnz || fread(s.val.data(), 8, nnz, f) != nnz
        || s.ptr.back() != (int)nnz) { fprintf(stderr, "invalid sparse matrix\n"); exit(1); }
    for (int i = 0; i < rows; i++)
        if (s.ptr[i] > s.ptr[i + 1]) { fprintf(stderr, "invalid sparse row pointers\n"); exit(1); }
    for (int c : s.col)
        if (c < 0 || c >= (int)cols) { fprintf(stderr, "invalid sparse column\n"); exit(1); }
}

static void finalize_model(Model& m) {
    for (auto& l : m.ly) {
        l.active = l.qkv.nnz || l.out.nnz || l.fi.nnz || l.fo.nnz;
        l.active_heads.resize(m.H, false);
        for (int h = 0; h < m.H; h++) {
            for (int row = 0; row < m.D && !l.active_heads[h]; row++)
                for (int k = l.out.ptr[row]; k < l.out.ptr[row + 1]; k++)
                    if (l.out.col[k] == 2 * h || l.out.col[k] == 2 * h + 1) {
                        l.active_heads[h] = true; break;
                    }
        }
        if (m.dense) {
            l.dense_qkv.build(l.qkv); l.dense_out.build(l.out);
            l.dense_fi.build(l.fi); l.dense_fo.build(l.fo);
        }
    }
    if (m.dense) { m.dense_emb.build(m.emb); m.dense_head.build(m.head); }
}

static void load(Model& m, const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    char magic[8];
    bool sparse = fread(magic, 1, 8, f) == 8 && memcmp(magic, "TVMSPAR\0", 8) == 0;
    if (sparse) {
        uint32_t version;
        if (fread(&version, 4, 1, f) != 1 || version != 1) { fprintf(stderr, "unsupported sparse model\n"); exit(1); }
    } else {
        rewind(f);
    }
    int32_t h[6];
    if (fread(h, 4, 6, f) != 6) { fprintf(stderr, "bad header\n"); exit(1); }
    m.V = h[0];
    m.D = h[1];
    m.L = h[2];
    m.H = h[3];
    m.F = h[4];
    m.stop = h[5];

    m.name.resize(m.V);
    for (int i = 0; i < m.V; i++) {
        uint32_t n;
        fread(&n, 4, 1, f);
        m.name[i].resize(n);
        fread(&m.name[i][0], 1, n, f);
        m.id[m.name[i]] = i;
    }

    int D = m.D, L = m.L, F = m.F, V = m.V;
    m.ly.resize(L);
    if (sparse) {
        read_sparse(f, m.emb);
        for (int i = 0; i < L; i++) {
            read_sparse(f, m.ly[i].qkv); read_sparse(f, m.ly[i].out);
            read_sparse(f, m.ly[i].fi); read_sparse(f, m.ly[i].fo);
        }
        read_sparse(f, m.head);
    } else {
        size_t per = (size_t)(3*D*D + D*D + 2*F*D + D*F);
        size_t tot = (size_t)V*D + L*per + (size_t)V*D;
        std::vector<double> w(tot);
        if (fread(w.data(), 8, tot, f) != tot) { fprintf(stderr, "truncated weight file\n"); exit(1); }
        const double* p = w.data();
        m.emb.build(p, V, D); p += V*D;
        for (int i = 0; i < L; i++) {
            auto& l = m.ly[i];
            l.qkv.build(p, 3*D, D); p += 3*D*D;
            l.out.build(p, D, D); p += D*D;
            l.fi.build(p, 2*F, D); p += 2*F*D;
            l.fo.build(p, D, F); p += D*F;
        }
        m.head.build(p, V, D);
    }
    int32_t has_erase = 0;
    if (fread(&has_erase, 4, 1, f) == 1 && has_erase) {
        m.attn_erase.resize(L);
        m.ffn_erase.resize(L);
        for (int l = 0; l < L; l++) {
            int32_t n;
            fread(&n, 4, 1, f);
            m.attn_erase[l].resize(n);
            for (int j = 0; j < n; j++) fread(&m.attn_erase[l][j], 4, 1, f);
            fread(&n, 4, 1, f);
            m.ffn_erase[l].resize(n);
            for (int j = 0; j < n; j++) fread(&m.ffn_erase[l][j], 4, 1, f);
        }
    }

    int32_t has_tb = 0;
    if (fread(&has_tb, 4, 1, f) == 1 && has_tb) {
        m.head_tb.resize(L);
        int H = m.H;
        for (int l = 0; l < L; l++) {
            m.head_tb[l].resize(H, TieBreak::AVERAGE);
            for (int h = 0; h < H; h++) {
                int32_t v;
                fread(&v, 4, 1, f);
                m.head_tb[l][h] = (v == 1) ? TieBreak::LATEST : TieBreak::AVERAGE;
            }
        }
    }
    int32_t has_ht = 0;
    if (fread(&has_ht, 4, 1, f) == 1 && has_ht) {
        m.head_type.resize(L, std::vector<int>(m.H));
        for (int l = 0; l < L; l++)
            for (int h = 0; h < m.H; h++)
                fread(&m.head_type[l][h], 4, 1, f);
    }
    fclose(f);

    finalize_model(m);
    printf("Head sparsity: %d/%d nonzero (%.0f%% sparse)\n",
            m.head.nnz, V * D, 100.0 * (1.0 - (double)m.head.nnz / (V * D)));
}

// ── Position encoding ──────────────────────────────────────────────────

static inline void add_position_encoding(double* x, int pos) {
    x[0] += pos;
    x[1] += 1.0 / std::log(2.0) - 1.0 / std::log((double)(pos + 2));
    x[2] += (double)pos * pos;
}

// ── Linear algebra ─────────────────────────────────────────────────────

static inline void matvec(const double* __restrict__ W,
                           const double* __restrict__ x,
                           double* __restrict__ y, int rows, int cols) {
#if defined(__APPLE__) && !defined(NO_BLAS)
    cblas_dgemv(CblasRowMajor, CblasNoTrans, rows, cols,
                1.0, W, cols, x, 1, 0.0, y, 1);
#else
    for (int i = 0; i < rows; i++) {
        double s = 0;
        const double* r = W + i * cols;
        for (int j = 0; j < cols; j++) s += r[j] * x[j];
        y[i] = s;
    }
#endif
}

static inline void matvec(const SparseMatrix& W, const double* x, double* y) {
    for (int i = 0; i < W.rows; i++) {
        double s = 0;
        for (int k = W.ptr[i]; k < W.ptr[i + 1]; k++) s += W.val[k] * x[W.col[k]];
        y[i] = s;
    }
}

static inline void matvec(const DenseMatrix& W, const double* x, double* y) {
    matvec(W.val.data(), x, y, W.rows, W.cols);
}

// ── Timing ─────────────────────────────────────────────────────────────

using Clock = std::chrono::steady_clock;
using TP = Clock::time_point;
static inline double secs(TP a, TP b) {
    return std::chrono::duration<double>(b - a).count();
}

// ── Main ───────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s model.bin [--dense] [--regen] [--output-hex] [--trace[=N]] [--brute|--nohull] [--args=STR] prog1.txt [...]\n", argv[0]);
        return 1;
    }

    bool regen = false, brute = false, dense = false, output_hex = false;
    int trace_every = 0;
    const char* trace_file = nullptr;
    const char* args_str = nullptr;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--regen") == 0) regen = true;
        if (strcmp(argv[i], "--dense") == 0) dense = true;
        if (strcmp(argv[i], "--output-hex") == 0) output_hex = true;
        if (strcmp(argv[i], "--brute") == 0 || strcmp(argv[i], "--nohull") == 0) brute = true;
        if (strncmp(argv[i], "--trace", 7) == 0) {
            trace_every = 1;
            if (argv[i][7] == '=') trace_every = atoi(argv[i] + 8);
        }
        if (strncmp(argv[i], "--trace-file=", 13) == 0) trace_file = argv[i] + 13;
        if (strncmp(argv[i], "--args=", 7) == 0) args_str = argv[i] + 7;
        else if (strcmp(argv[i], "--args") == 0 && i + 1 < argc) args_str = argv[++i];
    }
    if (brute) printf("Using brute-force O(n) KV cache\n");

    Model m;
    m.dense = dense;
    load(m, argv[1]);
    if (dense) printf("Using dense projection benchmark mode\n");
    printf("Loaded: vocab=%d D=%d layers=%d heads=%d d_ffn=%d\n",
           m.V, m.D, m.L, m.H, m.F);

    int D = m.D, F = m.F, H = m.H, L = m.L;
    int passed = 0, failed = 0, skipped = 0;
    long total_tok = 0, total_ops = 0;
    double total_time = 0, t_proj = 0, t_hull = 0, t_head = 0;

    for (int ai = 2; ai < argc; ai++) {
        if (strstr(argv[ai], "_ref")) continue;
        if (argv[ai][0] == '-' && argv[ai][1] == '-') continue;
        const char* pf = argv[ai];

        const char* bn = strrchr(pf, '/');
        bn = bn ? bn + 1 : pf;
        std::string test(bn);
        auto dot = test.rfind('.');
        if (dot != std::string::npos) test.resize(dot);

        std::ifstream fin(pf);
        if (!fin) { fprintf(stderr, "cannot open %s\n", pf); continue; }
        std::vector<int> ids;
        std::string tok;
        while (fin >> tok) {
            auto it = m.id.find(tok);
            if (it == m.id.end()) {
                fprintf(stderr, "unknown token '%s' in %s\n", tok.c_str(), pf);
                exit(1);
            }
            ids.push_back(it->second);
        }
        if (args_str) {
            int close_pos = -1;
            for (int i = (int)ids.size() - 1; i >= 0; i--)
                if (m.name[ids[i]] == "}") { close_pos = i; break; }
            if (close_pos >= 0)
                ids.resize(close_pos + 1);
            for (int i = 0; args_str[i]; i++) {
                std::string ch(1, args_str[i]);
                auto it = m.id.find(ch);
                if (it == m.id.end()) {
                    fprintf(stderr, "warning: char '%c' not in vocabulary\n", args_str[i]);
                } else {
                    ids.push_back(it->second);
                }
            }
            auto it00 = m.id.find("00");
            if (it00 != m.id.end()) ids.push_back(it00->second);
        }
        int plen = (int)ids.size();

        std::string rp(pf);
        auto dp = rp.rfind('.');
        if (dp != std::string::npos) rp.insert(dp, "_ref");
        int max_gen = MAX_GEN;
        if (!regen) {
            std::ifstream rf(rp);
            if (rf) {
                int ref_count = 0;
                std::string rt;
                while (rf >> rt) ref_count++;
                max_gen = ref_count - plen + 100;
                if (max_gen < 100) max_gen = 100;
            }
        }
        ids.reserve(plen + max_gen);

        std::vector<HardAttentionHead> hulls(brute ? 0 : L * H);
        std::vector<BruteAttentionHead> brutes(brute ? L * H : 0);
        std::vector<std::vector<double>> gather_values(L * H);
        size_t cache_entries = 0;
        int seq = 0;

        std::vector<double> x(D), qkv(3*D), ho(D), ao(D),
                            ff(2*F), gv(F), fo(D), logits(m.V);

        printf("%s: ", test.c_str()); fflush(stdout);
        auto t0 = Clock::now();
        bool halted = false;
        bool trapped = false;

        for (int pos = 0; pos < plen + max_gen; pos++) {
            if (dense) {
                const double* e = m.dense_emb.val.data() + (size_t)ids[pos] * D;
                std::copy(e, e + D, x.data());
            } else {
                std::fill(x.begin(), x.end(), 0.0);
                for (int k = m.emb.ptr[ids[pos]]; k < m.emb.ptr[ids[pos] + 1]; k++)
                    x[m.emb.col[k]] = m.emb.val[k];
            }
            add_position_encoding(x.data(), pos);

            for (int l = 0; l < L; l++) {
                const auto& ly = m.ly[l];
                if (!dense && !ly.active) continue;

                auto ta = Clock::now();
                if (dense) matvec(ly.dense_qkv, x.data(), qkv.data());
                else matvec(ly.qkv, x.data(), qkv.data());
                auto tb = Clock::now();

                double *q = qkv.data(), *k = q + D, *v = k + D;
                int base = l * H;
                for (int h = 0; h < H; h++) {
                    if (!dense && !ly.active_heads[h]) { ho[h * 2] = ho[h * 2 + 1] = 0.0; continue; }
                    int type = !m.head_type.empty() ? m.head_type[l][h] : 0;
                    if (!brute && type == 1) {
                        ho[h * 2] = v[h * 2];
                        ho[h * 2 + 1] = v[h * 2 + 1];
                        continue;
                    }
                    if (!brute && type == 2) {
                        auto& values = gather_values[base + h];
                        values.push_back(v[h * 2]);
                        values.push_back(v[h * 2 + 1]);
                        cache_entries++;
                        int idx = q[h * 2 + 1] == 0.0 ? 0 : (int)std::round(q[h * 2] / q[h * 2 + 1]);
                        idx = std::max(0, std::min(idx, pos));
                        ho[h * 2] = values[2 * idx];
                        ho[h * 2 + 1] = values[2 * idx + 1];
                        continue;
                    }
                    TieBreak tb = (!m.head_tb.empty()) ? m.head_tb[l][h] : TieBreak::AVERAGE;
                    if (brute) {
                        brutes[base+h].insert(&k[h*2], &v[h*2], seq);
                        cache_entries++;
                        brutes[base+h].query(&q[h*2], tb, &ho[h*2]);
                    } else {
                        hulls[base+h].insert(&k[h*2], &v[h*2], seq);
                        cache_entries++;
                        hulls[base+h].query(&q[h*2], tb, &ho[h*2]);
                    }
                }
                seq++;
                auto tc = Clock::now();

                if (dense) matvec(ly.dense_out, ho.data(), ao.data());
                else matvec(ly.out, ho.data(), ao.data());
                for (int i = 0; i < D; i++) x[i] += ao[i];
                if (dense) matvec(ly.dense_fi, x.data(), ff.data());
                else matvec(ly.fi, x.data(), ff.data());
                for (int i = 0; i < F; i++)
                    gv[i] = (ff[i] > 0 ? ff[i] : 0.0) * ff[F + i];
                if (dense) matvec(ly.dense_fo, gv.data(), fo.data());
                else matvec(ly.fo, gv.data(), fo.data());
                for (int i = 0; i < D; i++) x[i] += fo[i];
                auto td = Clock::now();

                t_proj += secs(ta, tb) + secs(tc, td);
                t_hull += secs(tb, tc);
            }

            if (pos + 1 == (int)ids.size()) {
                auto te = Clock::now();
                const auto& sp = m.head;
                int best = 0; double bs = -1e300;
                for (int i = 0; i < sp.rows; i++) {
                    double s = 0;
                    if (dense) {
                        const double* row = m.dense_head.val.data() + (size_t)i * D;
                        for (int j = 0; j < D; j++) s += row[j] * x[j];
                    } else {
                        for (int k = sp.ptr[i]; k < sp.ptr[i + 1]; k++) s += sp.val[k] * x[sp.col[k]];
                    }
                    if (s > bs) { bs = s; best = i; }
                }
                auto tf = Clock::now();
                t_head += secs(te, tf);
                ids.push_back(best);

                if (trace_every > 0) {
                    int gen = pos + 1 - plen;
                    if (gen % trace_every == 0 || best == m.stop) {
                        double elapsed = secs(t0, tf);
                        fprintf(stderr, "[%7d %7.3fs %6.0f tok/s] %s\n",
                                gen, elapsed,
                                gen > 0 ? gen / elapsed : 0.0,
                                m.name[best].c_str());
                    }
                }

                if (m.name[best] == "halt" || m.name[best] == "trap") {
                    halted = m.name[best] == "halt";
                    trapped = m.name[best] == "trap";
                    break;
                }
            }
        }

        auto t1 = Clock::now();
        double dt = secs(t0, t1);
        int nt = (int)ids.size(), no = 0;
        std::string output_bytes;
        for (int i : ids) {
            const auto& s = m.name[i];
            if (s.find("commit") != std::string::npos || s == "branch_taken")
                no++;
            if (s.size() > 4 && s[0] == 'o' && s[1] == 'u' && s[2] == 't' && s[3] == '(') {
                unsigned byte_val;
                if (s.size() == 6 && s[5] == ')') {
                    byte_val = (unsigned char)s[4];
                } else {
                    byte_val = 0;
                    for (int j = 4; j < (int)s.size() && s[j] != ')'; j++) {
                        int d = (s[j] >= 'a') ? s[j] - 'a' + 10 : s[j] - '0';
                        byte_val = byte_val * 16 + d;
                    }
                }
                output_bytes.push_back((char)byte_val);
            }
        }
        total_tok += nt;
        total_ops += no;
        total_time += dt;

        if (trace_file) {
            FILE* out = fopen(trace_file, "wb");
            if (!out) { fprintf(stderr, "cannot write %s\n", trace_file); return 1; }
            for (int id : ids) {
                const auto& token = m.name[id];
                fprintf(out, "%zu:", token.size());
                fwrite(token.data(), 1, token.size(), out);
                fputc('\n', out);
            }
            fclose(out);
        }

        if (trapped) {
            printf("FAIL execution trapped (%.2fs)\n", dt);
            failed++;
        } else if (!halted) {
            printf("FAIL generation limit exhausted without halt (%.2fs)\n", dt);
            failed++;
        } else if (regen) {
            FILE* out = fopen(rp.c_str(), "w");
            if (!out) { fprintf(stderr, "cannot write %s\n", rp.c_str()); continue; }
            bool in_prog = false;
            int prog_count = 0;
            std::vector<std::string> line_buf;
            for (int i = 0; i < nt; i++) {
                const auto& s = m.name[ids[i]];
                const char* ws = (s == " ") ? "<sp>" : s.c_str();
                if (s == "{") { fprintf(out, "{\n"); in_prog = true; prog_count = 0; continue; }
                if (s == "}") { fprintf(out, "}\n"); in_prog = false; continue; }
                if (in_prog) {
                    if (prog_count > 0) fputc(' ', out);
                    fputs(ws, out);
                    if (++prog_count == 5) { fputc('\n', out); prog_count = 0; }
                    continue;
                }
                if (!line_buf.empty()) fputc(' ', out);
                fputs(ws, out);
                line_buf.push_back(s);
                bool terminal = s.find("commit") != std::string::npos
                    || s.rfind("out(", 0) == 0 || s == "halt"
                    || s == "branch_taken" || s == "call_commit"
                    || s == "return_commit";
                if (terminal) { fputc('\n', out); line_buf.clear(); }
            }
            if (!line_buf.empty()) fputc('\n', out);
            fclose(out);
            printf("REGEN %d tok, %d ops in %.2fs (%.0f tok/s)\n",
                   nt, no, dt, dt > 0 ? nt/dt : 0.0);
            passed++;
        } else {
            std::ifstream rf(rp);
            if (rf) {
                std::vector<std::string> ref;
                std::string rt;
                while (rf >> rt) {
                    if (rt == "<sp>") rt = " ";
                    ref.push_back(rt);
                }
                int mm = -1, mn = std::min(nt, (int)ref.size());
                for (int i = 0; i < mn; i++)
                    if (m.name[ids[i]] != ref[i]) { mm = i; break; }
                if (mm >= 0) {
                    printf("  MISMATCH at %d: predicted=%s, expected=%s\n",
                           mm, m.name[ids[mm]].c_str(), ref[mm].c_str());
                    printf("FAIL (%.2fs)\n", dt);
                    failed++;
                } else if (nt != (int)ref.size()) {
                    printf("  MISMATCH: generated %d tokens, expected %d\n", nt, (int)ref.size());
                    printf("FAIL (%.2fs)\n", dt);
                    failed++;
                } else {
                    printf("PASS  %d tok, %d ops in %.2fs (%.0f tok/s)\n",
                           nt, no, dt, dt > 0 ? nt/dt : 0.0);
                    passed++;
                }
            } else {
                printf("RAN   %d tok, %d ops in %.2fs (%.0f tok/s)\n",
                       nt, no, dt, dt > 0 ? nt/dt : 0.0);
                skipped++;
            }
            if (!output_bytes.empty()) {
                printf("  %s: ", output_hex ? "output_hex" : "output");
                for (unsigned char c : output_bytes) {
                    if (output_hex)
                        printf("%02x", c);
                    else
                        putchar((c >= 0x20 && c < 0x7f) || c == '\n' || c == '\t' ? c : '.');
                }
                putchar('\n');
            }
            printf("  peak cache: %zu logical head entries\n", cache_entries);
        }
    }

    printf("\n%d passed, %d failed, %d no-ref\n", passed, failed, skipped);
    if (total_time > 0) {
        double t_misc = total_time - t_proj - t_hull - t_head;
        printf("\nBenchmark: %ld tok, %ld ops, %.2fs\n"
               "  %.0f tok/s, %.0f wasm-ops/s\n",
               total_tok, total_ops, total_time,
               total_tok / total_time,
               total_ops > 0 ? total_ops / total_time : 0.0);
        printf("\nTime breakdown:\n"
               "  proj:  %.3fs (%4.1f%%)\n"
               "  hull:  %.3fs (%4.1f%%)\n"
               "  head:  %.3fs (%4.1f%%)\n"
               "  misc:  %.3fs (%4.1f%%)\n",
               t_proj, 100*t_proj/total_time,
               t_hull, 100*t_hull/total_time,
               t_head, 100*t_head/total_time,
               t_misc, 100*t_misc/total_time);
    }
    return failed ? 1 : 0;
}

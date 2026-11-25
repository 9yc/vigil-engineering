#include "transformer_ops.hpp"

#include <sys/stat.h>

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct CliArgs {
  std::map<std::string, std::string> opts;
  std::set<std::string> flags;

  bool parse(int argc, char** argv, const std::set<std::string>& known_opts,
             const std::set<std::string>& known_flags) {
    for (int i = 2; i < argc; ++i) {
      const std::string a = argv[i];
      if (known_flags.count(a)) {
        flags.insert(a);
      } else if (known_opts.count(a) && i + 1 < argc) {
        opts[a] = argv[++i];
      } else {
        std::cerr << "Error: unknown or incomplete option '" << a << "'\n";
        return false;
      }
    }
    return true;
  }

  std::string str(const std::string& k, const std::string& def = "") const {
    auto it = opts.find(k);
    return it == opts.end() ? def : it->second;
  }
  uint32_t u32(const std::string& k, uint32_t def = 0) const {
    auto it = opts.find(k);
    return it == opts.end() ? def : static_cast<uint32_t>(std::stoul(it->second));
  }
  uint64_t u64(const std::string& k, uint64_t def = 0) const {
    auto it = opts.find(k);
    return it == opts.end() ? def : std::stoull(it->second);
  }
  float f32(const std::string& k, float def = 0.0f) const {
    auto it = opts.find(k);
    return it == opts.end() ? def : std::stof(it->second);
  }
  bool has(const std::string& k) const { return flags.count(k) > 0; }
};

bool file_exists(const std::string& path) {
  struct stat st{};
  return ::stat(path.c_str(), &st) == 0;
}

bool ensure_output_writable(const std::string& path, bool overwrite) {
  if (path.empty()) return false;
  if (!overwrite && file_exists(path)) {
    std::cerr << "Error: output file already exists (pass --overwrite to replace): " << path << "\n";
    return false;
  }
  return true;
}

bool write_text_file(const std::string& path, const std::string& content, bool overwrite) {
  if (!ensure_output_writable(path, overwrite)) return false;
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out.is_open()) return false;
  out.write(content.data(), static_cast<std::streamsize>(content.size()));
  out.flush();
  return out.good();
}

bool read_f32_binary(const std::string& path, size_t expected_count, std::vector<float>* out) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in.is_open()) return false;
  const std::streamsize sz = in.tellg();
  if (sz < 0 || static_cast<size_t>(sz) != expected_count * sizeof(float)) return false;
  in.seekg(0, std::ios::beg);
  std::vector<uint8_t> raw(static_cast<size_t>(sz));
  if (sz > 0) {
    in.read(reinterpret_cast<char*>(raw.data()), sz);
    if (in.gcount() != sz) return false;
  }
  out->resize(expected_count);
  for (size_t i = 0; i < expected_count; ++i) (*out)[i] = rt_ops::read_f32_le(raw.data() + 4u * i);
  return true;
}

std::vector<float> make_synthetic_f32(size_t count, uint64_t seed, float scale = 0.25f) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<float> dist(0.0f, scale);
  std::vector<float> out(count);
  for (size_t i = 0; i < count; ++i) out[i] = dist(rng);
  return out;
}

std::vector<int32_t> parse_int_csv(const std::string& csv) {
  std::vector<int32_t> res;
  std::stringstream ss(csv);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (!item.empty()) res.push_back(static_cast<int32_t>(std::stol(item)));
  }
  return res;
}

std::string format_f32_array(const std::vector<float>& vals) {
  std::ostringstream oss;
  oss << std::setprecision(17) << "[";
  for (size_t i = 0; i < vals.size(); ++i) {
    if (i > 0) oss << ", ";
    oss << static_cast<double>(vals[i]);
  }
  oss << "]";
  return oss.str();
}

int cmd_pack_emb(int argc, char** argv) {
  CliArgs a;
  if (!a.parse(argc, argv, {"--input", "--output", "--stats-json", "--vocab", "--hidden", "--seed"},
               {"--synthetic", "--overwrite"})) {
    return 2;
  }
  const uint32_t vocab = a.u32("--vocab"), hidden = a.u32("--hidden");
  const std::string input_path = a.str("--input"), output_path = a.str("--output");
  const std::string stats_path = a.str("--stats-json");
  const bool synthetic = a.has("--synthetic"), overwrite = a.has("--overwrite");
  if (vocab == 0 || hidden == 0 || output_path.empty() || (!synthetic && input_path.empty())) return 2;
  if (!ensure_output_writable(output_path, overwrite)) return 1;
  if (!stats_path.empty() && !ensure_output_writable(stats_path, overwrite)) return 1;

  const size_t total = static_cast<size_t>(vocab) * hidden;
  std::vector<float> weights;
  if (synthetic) {
    weights = make_synthetic_f32(total, a.u64("--seed", 42), 0.5f);
  } else if (!read_f32_binary(input_path, total, &weights)) {
    std::cerr << "Error: failed to read float32 input from " << input_path << "\n";
    return 1;
  }

  rt_ops::QuantAuditStats stats{};
  if (!rt_ops::write_emb1_file(output_path, weights.data(), vocab, hidden, &stats)) return 1;

  std::ostringstream oss;
  oss << std::setprecision(9) << "{\n"
      << "  \"magic\": \"EMB1\",\n  \"vocab_size\": " << vocab << ",\n"
      << "  \"hidden_size\": " << hidden << ",\n  \"row_stride\": " << (4u + hidden) << ",\n"
      << "  \"cos_min\": " << stats.cos_min << ",\n  \"cos_mean\": " << stats.cos_mean << ",\n"
      << "  \"max_abs_err\": " << stats.max_abs_err << "\n}\n";
  if (!stats_path.empty()) return write_text_file(stats_path, oss.str(), overwrite) ? 0 : 1;
  std::cout << oss.str();
  return 0;
}

int cmd_lookup_emb(int argc, char** argv) {
  CliArgs a;
  if (!a.parse(argc, argv, {"--emb", "--format", "--tokens", "--hidden", "--output"}, {"--overwrite"})) {
    return 2;
  }
  const std::string emb_path = a.str("--emb"), format = a.str("--format", "emb1");
  const std::string tokens_csv = a.str("--tokens"), output_path = a.str("--output");
  const uint32_t hidden = a.u32("--hidden");
  const bool overwrite = a.has("--overwrite");
  if (emb_path.empty() || tokens_csv.empty()) return 2;
  if (!output_path.empty() && !ensure_output_writable(output_path, overwrite)) return 1;

  const std::vector<int32_t> tokens = parse_int_csv(tokens_csv);
  if (tokens.empty()) return 2;

  uint32_t out_hidden = hidden, out_vocab = 0;
  std::vector<std::vector<float>> rows;
  if (format == "emb1") {
    rt_ops::EmbI8Reader reader;
    if (reader.open(emb_path, hidden) != 0) return 1;
    out_hidden = reader.hidden_size();
    out_vocab = reader.vocab_size();
    for (int32_t tok : tokens) {
      std::vector<float> vec(out_hidden);
      if (reader.lookup_f32(tok, vec.data()) != 0) return 1;
      rows.push_back(std::move(vec));
    }
  } else if (format == "bf16") {
    if (hidden == 0) return 2;
    for (int32_t tok : tokens) {
      std::vector<float> vec(hidden);
      if (rt_ops::EmbI8Reader::lookup_bf16_raw_file(emb_path, hidden, tok, vec.data()) != 0) return 1;
      rows.push_back(std::move(vec));
    }
  } else {
    return 2;
  }

  std::ostringstream oss;
  oss << "{\n  \"format\": \"" << format << "\",\n  \"vocab_size\": " << out_vocab
      << ",\n  \"hidden_size\": " << out_hidden << ",\n  \"tokens\": [";
  for (size_t i = 0; i < tokens.size(); ++i) oss << (i ? ", " : "") << tokens[i];
  oss << "],\n  \"rows\": [\n";
  for (size_t i = 0; i < rows.size(); ++i) {
    oss << "    " << format_f32_array(rows[i]) << (i + 1u < rows.size() ? "," : "") << "\n";
  }
  oss << "  ]\n}\n";
  if (!output_path.empty()) return write_text_file(output_path, oss.str(), overwrite) ? 0 : 1;
  std::cout << oss.str();
  return 0;
}

int cmd_pack_lmhead(int argc, char** argv) {
  CliArgs a;
  if (!a.parse(argc, argv, {"--input", "--norm-weight", "--output", "--vocab", "--hidden", "--rms-eps", "--seed"},
               {"--synthetic", "--overwrite"})) {
    return 2;
  }
  const uint32_t vocab = a.u32("--vocab"), hidden = a.u32("--hidden");
  const std::string input_path = a.str("--input"), norm_path = a.str("--norm-weight");
  const std::string output_path = a.str("--output");
  const bool synthetic = a.has("--synthetic"), overwrite = a.has("--overwrite");
  if (vocab == 0 || hidden == 0 || output_path.empty() || (!synthetic && input_path.empty())) return 2;
  if (!ensure_output_writable(output_path, overwrite)) return 1;

  std::vector<float> weights, norm_w;
  if (synthetic) {
    weights = make_synthetic_f32(static_cast<size_t>(vocab) * hidden, a.u64("--seed", 42), 0.25f);
  } else if (!read_f32_binary(input_path, static_cast<size_t>(vocab) * hidden, &weights)) {
    return 1;
  }
  const float* norm_ptr = nullptr;
  if (!norm_path.empty()) {
    if (!read_f32_binary(norm_path, hidden, &norm_w)) return 1;
    norm_ptr = norm_w.data();
  }
  return rt_ops::write_lmh1_file(
             output_path, weights.data(), vocab, hidden, norm_ptr, a.f32("--rms-eps", 1e-6f))
             ? 0
             : 1;
}

int cmd_run_lmhead(int argc, char** argv) {
  CliArgs a;
  if (!a.parse(argc, argv,
               {"--lmhead", "--hidden-in", "--norm-weight", "--output", "--threads", "--top-k",
                "--top-p", "--temp", "--seed", "--draws"},
               {"--no-norm", "--synthetic-input", "--overwrite"})) {
    return 2;
  }
  const std::string lmhead_path = a.str("--lmhead"), hidden_in_path = a.str("--hidden-in");
  const std::string output_path = a.str("--output");
  const bool synthetic_in = a.has("--synthetic-input"), overwrite = a.has("--overwrite");
  if (lmhead_path.empty() || (!synthetic_in && hidden_in_path.empty())) return 2;
  if (!output_path.empty() && !ensure_output_writable(output_path, overwrite)) return 1;

  rt_ops::QuantLmHead head;
  if (head.load(lmhead_path, a.str("--norm-weight")) != 0) return 1;

  const uint64_t seed = a.u64("--seed", 12345);
  std::vector<float> x_in;
  if (synthetic_in) {
    x_in = make_synthetic_f32(head.hidden_size(), seed ^ 0x9e3779b97f4a7c15ULL, 0.5f);
  } else if (!read_f32_binary(hidden_in_path, head.hidden_size(), &x_in)) {
    return 1;
  }

  rt_ops::WorkerPool pool(a.u32("--threads", 1));
  std::vector<float> logits(head.vocab_size());
  const int32_t argmax_id = head.forward(pool, x_in.data(), logits.data(), !a.has("--no-norm"));
  if (argmax_id < 0) return 1;

  const rt_ops::SamplerConfig scfg{a.f32("--temp", 1.0f), a.u32("--top-k", 0), a.f32("--top-p", 1.0f)};
  const auto dist = rt_ops::build_sampling_distribution(logits.data(), head.vocab_size(), scfg);
  std::mt19937_64 rng(seed);
  const uint32_t draws = a.u32("--draws", 8);
  std::vector<int32_t> samples(draws);
  for (uint32_t i = 0; i < draws; ++i) {
    samples[i] = rt_ops::sample_from_logits(logits.data(), head.vocab_size(), scfg, rng);
  }

  std::ostringstream oss;
  oss << std::setprecision(9) << "{\n"
      << "  \"vocab_size\": " << head.vocab_size() << ",\n  \"hidden_size\": " << head.hidden_size()
      << ",\n  \"threads\": " << pool.num_threads() << ",\n  \"argmax\": " << argmax_id
      << ",\n  \"logits\": " << format_f32_array(logits) << ",\n  \"distribution\": [\n";
  for (size_t i = 0; i < dist.size(); ++i) {
    oss << "    {\"token_id\": " << dist[i].token_id << ", \"prob\": " << std::setprecision(12)
        << dist[i].prob << "}" << (i + 1u < dist.size() ? "," : "") << "\n";
  }
  oss << "  ],\n  \"samples\": [";
  for (size_t i = 0; i < samples.size(); ++i) oss << (i ? ", " : "") << samples[i];
  oss << "]\n}\n";

  if (!output_path.empty()) return write_text_file(output_path, oss.str(), overwrite) ? 0 : 1;
  std::cout << oss.str();
  return 0;
}

int cmd_verify_block(int argc, char** argv) {
  CliArgs a;
  if (!a.parse(argc, argv,
               {"--weights", "--input", "--output", "--hidden", "--heads", "--kv-heads",
                "--head-dim", "--intermediate", "--seq-len", "--rms-eps", "--rope-theta", "--seed"},
               {"--w8a16", "--overwrite"})) {
    return 2;
  }
  const std::string output_path = a.str("--output");
  const uint32_t seq_len = a.u32("--seq-len", 4);
  const bool overwrite = a.has("--overwrite");
  if (output_path.empty() || seq_len == 0) return 2;
  if (!ensure_output_writable(output_path, overwrite)) return 1;

  rt_ops::BlockConfig cfg{};
  cfg.hidden_size = a.u32("--hidden", 64);
  cfg.n_heads = a.u32("--heads", 4);
  cfg.n_kv_heads = a.u32("--kv-heads", 2);
  cfg.head_dim = a.u32("--head-dim", 16);
  cfg.intermediate_size = a.u32("--intermediate", 128);
  cfg.max_seq_len = std::max(cfg.max_seq_len, seq_len);
  cfg.rms_eps = a.f32("--rms-eps", 1e-6f);
  cfg.rope_theta = a.f32("--rope-theta", 10000.0f);
  cfg.w8a16_quant = a.has("--w8a16");

  const size_t H = cfg.hidden_size, QD = static_cast<size_t>(cfg.n_heads) * cfg.head_dim;
  const size_t KVD = static_cast<size_t>(cfg.n_kv_heads) * cfg.head_dim, I = cfg.intermediate_size;
  const size_t total_w = 2u * H + QD * H + 2u * KVD * H + H * QD + 3u * I * H;
  const uint64_t seed = a.u64("--seed", 2026);

  std::vector<float> flat_w;
  if (!a.str("--weights").empty()) {
    if (!read_f32_binary(a.str("--weights"), total_w, &flat_w)) return 1;
  } else {
    flat_w = make_synthetic_f32(total_w, seed, 0.15f);
    for (size_t i = 0; i < 2u * H; ++i) flat_w[i] = 1.0f + 0.1f * flat_w[i];
  }

  rt_ops::BlockWeights bw{};
  size_t off = 0;
  auto slice = [&](size_t n) {
    std::vector<float> v(flat_w.begin() + static_cast<std::ptrdiff_t>(off),
                         flat_w.begin() + static_cast<std::ptrdiff_t>(off + n));
    off += n;
    return v;
  };
  bw.rms_attn = slice(H);
  bw.rms_mlp = slice(H);
  bw.w_q = slice(QD * H);
  bw.w_k = slice(KVD * H);
  bw.w_v = slice(KVD * H);
  bw.w_o = slice(H * QD);
  bw.w_gate = slice(I * H);
  bw.w_up = slice(I * H);
  bw.w_down = slice(H * I);

  std::vector<float> seq_in;
  if (!a.str("--input").empty()) {
    if (!read_f32_binary(a.str("--input"), static_cast<size_t>(seq_len) * H, &seq_in)) return 1;
  } else {
    seq_in = make_synthetic_f32(static_cast<size_t>(seq_len) * H, seed + 17ULL, 0.5f);
  }

  try {
    rt_ops::TransformerBlockRef block(cfg, std::move(bw));
    rt_ops::SingleSeqKvCache cache;
    cache.init(cfg.max_seq_len, cfg.n_kv_heads, cfg.head_dim);
    std::vector<std::vector<float>> outputs(seq_len, std::vector<float>(H));
    for (uint32_t t = 0; t < seq_len; ++t) {
      block.forward_step(seq_in.data() + static_cast<size_t>(t) * H, cache, outputs[t].data());
    }
    std::ostringstream oss;
    oss << "{\n  \"hidden_size\": " << cfg.hidden_size << ",\n  \"n_heads\": " << cfg.n_heads
        << ",\n  \"n_kv_heads\": " << cfg.n_kv_heads << ",\n  \"head_dim\": " << cfg.head_dim
        << ",\n  \"intermediate_size\": " << cfg.intermediate_size << ",\n  \"seq_len\": " << seq_len
        << ",\n  \"w8a16_quant\": " << (cfg.w8a16_quant ? "true" : "false")
        << ",\n  \"kv_cached_tokens\": " << cache.seq_len << ",\n  \"outputs\": [\n";
    for (uint32_t t = 0; t < seq_len; ++t) {
      oss << "    " << format_f32_array(outputs[t]) << (t + 1u < seq_len ? "," : "") << "\n";
    }
    oss << "  ]\n}\n";
    return write_text_file(output_path, oss.str(), overwrite) ? 0 : 1;
  } catch (const std::exception& ex) {
    std::cerr << "Error: " << ex.what() << "\n";
    return 1;
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage: transformer_ref_cli <pack-emb|lookup-emb|pack-lmhead|run-lmhead|verify-block>\n";
    return 2;
  }
  const std::string sub = argv[1];
  if (sub == "pack-emb") return cmd_pack_emb(argc, argv);
  if (sub == "lookup-emb") return cmd_lookup_emb(argc, argv);
  if (sub == "pack-lmhead") return cmd_pack_lmhead(argc, argv);
  if (sub == "run-lmhead") return cmd_run_lmhead(argc, argv);
  if (sub == "verify-block") return cmd_verify_block(argc, argv);
  std::cerr << "Error: unknown subcommand '" << sub << "'\n";
  return 2;
}

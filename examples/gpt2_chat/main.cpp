// Minimal GPT-2 chat demo on top of frugally-deep.
//
// Usage:
//   gpt2_chat <weights.bin> <vocab.json> <merges.txt> [max_new_tokens] [temperature] [top_k]
//
// Reads a prompt per line from stdin and prints the continuation. A blank
// line exits. Uses a stateful key/value cache (see
// include/fdeep/llm/gpt2_cached.hpp), so each new token costs roughly
// constant time rather than re-encoding the whole prefix.
//
// The weights binary is produced by
//   keras_export/save_gpt2_weights_bin.py --output weights.bin

#include <fdeep/llm/gpt2_bpe.hpp>
#include <fdeep/llm/gpt2_cached.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

int sample_logits(const std::vector<float>& logits, float temperature,
    std::size_t top_k, std::mt19937_64& rng)
{
    if (temperature <= 0.0f) {
        std::size_t best = 0;
        float best_val = logits[0];
        for (std::size_t i = 1; i < logits.size(); ++i) {
            if (logits[i] > best_val) {
                best_val = logits[i];
                best = i;
            }
        }
        return static_cast<int>(best);
    }

    std::vector<std::pair<float, int>> cand;
    cand.reserve(logits.size());
    const float inv_t = 1.0f / temperature;
    for (std::size_t i = 0; i < logits.size(); ++i) {
        cand.emplace_back(logits[i] * inv_t, static_cast<int>(i));
    }
    if (top_k > 0 && top_k < cand.size()) {
        std::partial_sort(cand.begin(),
            cand.begin() + static_cast<std::ptrdiff_t>(top_k), cand.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });
        cand.resize(top_k);
    }
    float m = cand[0].first;
    for (const auto& c : cand) m = std::max(m, c.first);
    float total = 0.0f;
    for (auto& c : cand) {
        c.first = std::exp(c.first - m);
        total += c.first;
    }
    std::uniform_real_distribution<float> dist(0.0f, total);
    const float pick = dist(rng);
    float acc = 0.0f;
    for (const auto& c : cand) {
        acc += c.first;
        if (acc >= pick) return c.second;
    }
    return cand.back().second;
}

}  // namespace

int main(int argc, char** argv)
{
    if (argc < 4) {
        std::cerr << "usage: " << argv[0]
                  << " <weights.bin> <vocab.json> <merges.txt>"
                     " [max_new_tokens] [temperature] [top_k] [max_seq_len]\n";
        return 1;
    }
    const std::string weights_path = argv[1];
    const std::string vocab_path = argv[2];
    const std::string merges_path = argv[3];
    const std::size_t max_new = (argc > 4) ? std::stoul(argv[4]) : 100;
    const float temperature = (argc > 5) ? std::stof(argv[5]) : 0.0f;
    const std::size_t top_k = (argc > 6) ? std::stoul(argv[6]) : 0;
    const std::size_t max_seq_len = (argc > 7) ? std::stoul(argv[7]) : 256;

    std::cerr << "loading tokenizer...\n";
    fdeep::llm::gpt2_bpe_tokenizer tok(vocab_path, merges_path);

    std::cerr << "loading weights...\n";
    const auto t0 = std::chrono::steady_clock::now();
    fdeep::llm::gpt2_cached_model gpt(weights_path, max_seq_len);
    const auto t1 = std::chrono::steady_clock::now();
    std::cerr << "loaded in "
              << std::chrono::duration<double>(t1 - t0).count() << " s\n";

    std::mt19937_64 rng(static_cast<std::uint64_t>(
        std::chrono::steady_clock::now().time_since_epoch().count()));

    std::cerr << "ready (max_seq_len=" << max_seq_len
              << ", temperature=" << temperature
              << ", top_k=" << top_k
              << "). type a prompt and press enter; blank line quits.\n";

    std::string line;
    while (true) {
        std::cout << "> " << std::flush;
        if (!std::getline(std::cin, line)) break;
        if (line.empty()) break;

        gpt.reset();
        const auto prompt_ids = tok.encode(line);
        if (prompt_ids.empty()) continue;
        if (prompt_ids.size() >= max_seq_len) {
            std::cerr << "[prompt is " << prompt_ids.size()
                      << " tokens; max_seq_len is " << max_seq_len
                      << "]\n";
            continue;
        }

        std::cout << line;
        std::cout.flush();

        const auto t_pre0 = std::chrono::steady_clock::now();
        auto logits = gpt.prefill(prompt_ids);
        const auto t_pre1 = std::chrono::steady_clock::now();

        std::vector<int> generated;
        const auto t_gen0 = std::chrono::steady_clock::now();
        for (std::size_t step = 0; step < max_new; ++step) {
            const int next_id = sample_logits(logits, temperature, top_k, rng);
            generated.push_back(next_id);
            std::cout << tok.decode({ next_id }) << std::flush;
            if (next_id == tok.eos_token_id()) break;
            if (gpt.cur_len() >= gpt.max_seq_len()) break;
            logits = gpt.step(next_id);
        }
        const auto t_gen1 = std::chrono::steady_clock::now();
        std::cout << "\n["
                  << "prefill " << prompt_ids.size() << " tok "
                  << std::chrono::duration<double>(t_pre1 - t_pre0).count() << " s, "
                  << "decode " << generated.size() << " tok "
                  << std::chrono::duration<double>(t_gen1 - t_gen0).count() << " s, "
                  << (generated.empty() ? 0.0 :
                          1000.0 * std::chrono::duration<double>(t_gen1 - t_gen0).count()
                          / static_cast<double>(generated.size()))
                  << " ms/tok]\n";
    }
    return 0;
}

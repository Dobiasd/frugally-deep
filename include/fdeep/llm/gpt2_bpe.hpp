// Copyright 2026, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

// GPT-2-compatible byte-level BPE tokenizer.
//
// Loads vocab.json (token-string -> id) and merges.txt (one merge rule per
// line, "<a> <b>") as exported from keras_hub's GPT2Tokenizer or HuggingFace's
// gpt2 tokenizer. Provides encode(text) -> vector<int> and decode(ids) -> string.
//
// Pre-tokenization uses GPT-2's regex pattern restricted to the ASCII subset.
// Non-ASCII text is byte-level safe (UTF-8 bytes round-trip via the byte
// encoder), but pre-tokenization will not split unicode words the way the
// reference Python implementation does. For the standard chat-demo use case
// this is sufficient.

#pragma once

#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fdeep {
namespace llm {

    namespace internal {

        // Build the canonical GPT-2 byte<->unicode mapping. Returns a 256-entry
        // table of UTF-8 strings, one per byte. The inverse mapping is built
        // alongside.
        inline std::array<std::string, 256> make_byte_to_unicode(
            std::unordered_map<std::string, uint8_t>& unicode_to_byte)
        {
            std::array<std::string, 256> table;
            std::vector<bool> mapped(256, false);

            auto encode_codepoint = [](uint32_t cp) {
                std::string out;
                if (cp < 0x80) {
                    out.push_back(static_cast<char>(cp));
                } else if (cp < 0x800) {
                    out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
                    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
                } else if (cp < 0x10000) {
                    out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
                    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
                    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
                } else {
                    out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
                    out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
                    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
                    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
                }
                return out;
            };

            // The "printable" ranges that map to themselves.
            auto add_range = [&](uint32_t lo, uint32_t hi) {
                for (uint32_t b = lo; b <= hi; ++b) {
                    table[b] = encode_codepoint(b);
                    mapped[b] = true;
                }
            };
            add_range(0x21, 0x7E);  // '!'..'~'
            add_range(0xA1, 0xAC);
            add_range(0xAE, 0xFF);

            // Remaining bytes get codepoints starting at 256, in byte order.
            uint32_t next_cp = 256;
            for (uint32_t b = 0; b < 256; ++b) {
                if (!mapped[b]) {
                    table[b] = encode_codepoint(next_cp++);
                    mapped[b] = true;
                }
            }

            unicode_to_byte.clear();
            for (uint32_t b = 0; b < 256; ++b) {
                unicode_to_byte[table[b]] = static_cast<uint8_t>(b);
            }
            return table;
        }

        // Strip surrounding whitespace from a string (in place).
        inline void strip(std::string& s)
        {
            std::size_t i = 0;
            while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\r' || s[i] == '\n')) {
                ++i;
            }
            std::size_t j = s.size();
            while (j > i && (s[j - 1] == ' ' || s[j - 1] == '\t' || s[j - 1] == '\r' || s[j - 1] == '\n')) {
                --j;
            }
            s = s.substr(i, j - i);
        }

        // Minimal JSON string-key -> int reader for vocab.json. Assumes the
        // file is the output of json.dump on a flat dict where keys are the
        // BPE token strings (already byte-encoded) and values are non-negative
        // integers. Handles standard JSON escapes (\\, \", \n, \t, \r, \b, \f,
        // \/) and \uXXXX surrogate pairs.
        inline std::unordered_map<std::string, int> load_vocab_json(const std::string& path)
        {
            std::ifstream in(path);
            if (!in) {
                throw std::runtime_error("could not open " + path);
            }
            std::ostringstream ss;
            ss << in.rdbuf();
            const std::string text = ss.str();

            std::unordered_map<std::string, int> vocab;
            vocab.reserve(60000);

            auto encode_codepoint = [](uint32_t cp, std::string& out) {
                if (cp < 0x80) {
                    out.push_back(static_cast<char>(cp));
                } else if (cp < 0x800) {
                    out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
                    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
                } else if (cp < 0x10000) {
                    out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
                    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
                    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
                } else {
                    out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
                    out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
                    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
                    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
                }
            };

            std::size_t i = 0;
            const std::size_t n = text.size();

            auto skip_ws = [&]() {
                while (i < n && (text[i] == ' ' || text[i] == '\t' || text[i] == '\n' || text[i] == '\r')) {
                    ++i;
                }
            };

            auto read_string = [&](std::string& out) {
                if (text[i] != '"') {
                    throw std::runtime_error("vocab.json: expected '\"' at offset " + std::to_string(i));
                }
                ++i;
                while (i < n && text[i] != '"') {
                    if (text[i] == '\\' && i + 1 < n) {
                        const char esc = text[++i];
                        switch (esc) {
                        case '"': out.push_back('"'); break;
                        case '\\': out.push_back('\\'); break;
                        case '/': out.push_back('/'); break;
                        case 'b': out.push_back('\b'); break;
                        case 'f': out.push_back('\f'); break;
                        case 'n': out.push_back('\n'); break;
                        case 'r': out.push_back('\r'); break;
                        case 't': out.push_back('\t'); break;
                        case 'u': {
                            if (i + 4 >= n) {
                                throw std::runtime_error("vocab.json: bad \\u escape");
                            }
                            uint32_t cp = 0;
                            for (int k = 0; k < 4; ++k) {
                                const char h = text[++i];
                                cp <<= 4;
                                if (h >= '0' && h <= '9') cp |= static_cast<uint32_t>(h - '0');
                                else if (h >= 'a' && h <= 'f') cp |= static_cast<uint32_t>(h - 'a' + 10);
                                else if (h >= 'A' && h <= 'F') cp |= static_cast<uint32_t>(h - 'A' + 10);
                                else throw std::runtime_error("vocab.json: bad hex digit");
                            }
                            if (cp >= 0xD800 && cp <= 0xDBFF && i + 6 < n
                                && text[i + 1] == '\\' && text[i + 2] == 'u') {
                                uint32_t lo = 0;
                                std::size_t j = i + 3;
                                for (int k = 0; k < 4; ++k) {
                                    const char h = text[j++];
                                    lo <<= 4;
                                    if (h >= '0' && h <= '9') lo |= static_cast<uint32_t>(h - '0');
                                    else if (h >= 'a' && h <= 'f') lo |= static_cast<uint32_t>(h - 'a' + 10);
                                    else if (h >= 'A' && h <= 'F') lo |= static_cast<uint32_t>(h - 'A' + 10);
                                    else throw std::runtime_error("vocab.json: bad hex digit");
                                }
                                if (lo >= 0xDC00 && lo <= 0xDFFF) {
                                    cp = 0x10000u + ((cp - 0xD800u) << 10) + (lo - 0xDC00u);
                                    i += 6;
                                }
                            }
                            encode_codepoint(cp, out);
                            break;
                        }
                        default:
                            throw std::runtime_error(
                                std::string("vocab.json: unknown escape \\") + esc);
                        }
                        ++i;
                    } else {
                        out.push_back(text[i++]);
                    }
                }
                if (i >= n) {
                    throw std::runtime_error("vocab.json: unterminated string");
                }
                ++i;  // consume closing quote
            };

            auto read_int = [&]() {
                std::size_t start = i;
                if (i < n && (text[i] == '-' || text[i] == '+')) ++i;
                while (i < n && text[i] >= '0' && text[i] <= '9') ++i;
                return std::stoi(text.substr(start, i - start));
            };

            skip_ws();
            if (i >= n || text[i] != '{') {
                throw std::runtime_error("vocab.json: expected '{'");
            }
            ++i;
            skip_ws();
            if (i < n && text[i] == '}') {
                return vocab;
            }
            while (i < n) {
                skip_ws();
                std::string key;
                read_string(key);
                skip_ws();
                if (i >= n || text[i] != ':') {
                    throw std::runtime_error("vocab.json: expected ':'");
                }
                ++i;
                skip_ws();
                vocab.emplace(std::move(key), read_int());
                skip_ws();
                if (i < n && text[i] == ',') {
                    ++i;
                    continue;
                }
                if (i < n && text[i] == '}') {
                    ++i;
                    break;
                }
                throw std::runtime_error("vocab.json: expected ',' or '}'");
            }
            return vocab;
        }

        inline std::vector<std::pair<std::string, std::string>> load_merges(const std::string& path)
        {
            std::ifstream in(path);
            if (!in) {
                throw std::runtime_error("could not open " + path);
            }
            std::vector<std::pair<std::string, std::string>> merges;
            std::string line;
            bool first = true;
            while (std::getline(in, line)) {
                if (first) {
                    first = false;
                    if (line.size() >= 1 && line[0] == '#') {
                        continue;  // skip "#version: 0.2" header
                    }
                }
                if (line.empty()) continue;
                const std::size_t sp = line.find(' ');
                if (sp == std::string::npos) continue;
                merges.emplace_back(line.substr(0, sp), line.substr(sp + 1));
            }
            return merges;
        }

    }  // namespace internal

    class gpt2_bpe_tokenizer {
    public:
        gpt2_bpe_tokenizer(const std::string& vocab_path, const std::string& merges_path)
            : byte_to_unicode_(internal::make_byte_to_unicode(unicode_to_byte_))
            , vocab_(internal::load_vocab_json(vocab_path))
        {
            const auto merges = internal::load_merges(merges_path);
            merge_ranks_.reserve(merges.size());
            for (std::size_t i = 0; i < merges.size(); ++i) {
                merge_ranks_.emplace(merges[i], static_cast<int>(i));
            }
            inv_vocab_.resize(vocab_.size());
            for (const auto& kv : vocab_) {
                if (kv.second < 0 || static_cast<std::size_t>(kv.second) >= inv_vocab_.size()) {
                    throw std::runtime_error("vocab id out of range");
                }
                inv_vocab_[static_cast<std::size_t>(kv.second)] = kv.first;
            }
        }

        std::size_t vocab_size() const { return vocab_.size(); }

        std::vector<int> encode(const std::string& text) const
        {
            std::vector<int> out;
            const auto pieces = pre_tokenize(text);
            for (const auto& piece : pieces) {
                std::string encoded;
                encoded.reserve(piece.size() * 2);
                for (unsigned char b : piece) {
                    encoded += byte_to_unicode_[b];
                }
                bpe_encode(encoded, out);
            }
            return out;
        }

        std::string decode(const std::vector<int>& ids) const
        {
            std::string concat;
            for (int id : ids) {
                if (id < 0 || static_cast<std::size_t>(id) >= inv_vocab_.size()) {
                    continue;  // skip unknown
                }
                concat += inv_vocab_[static_cast<std::size_t>(id)];
            }
            // Decode the byte-encoded unicode string back to raw bytes.
            std::string out;
            out.reserve(concat.size());
            std::size_t i = 0;
            while (i < concat.size()) {
                const unsigned char c = static_cast<unsigned char>(concat[i]);
                std::size_t len = 1;
                if ((c & 0x80) == 0) {
                    len = 1;
                } else if ((c & 0xE0) == 0xC0) {
                    len = 2;
                } else if ((c & 0xF0) == 0xE0) {
                    len = 3;
                } else if ((c & 0xF8) == 0xF0) {
                    len = 4;
                }
                if (i + len > concat.size()) break;
                const std::string ch = concat.substr(i, len);
                auto it = unicode_to_byte_.find(ch);
                if (it != unicode_to_byte_.end()) {
                    out.push_back(static_cast<char>(it->second));
                } else {
                    out += ch;
                }
                i += len;
            }
            return out;
        }

        int eos_token_id() const
        {
            auto it = vocab_.find("<|endoftext|>");
            return it != vocab_.end() ? it->second : -1;
        }

    private:
        // Apply GPT-2 pre-tokenization to ASCII input. Splits the text into
        // pieces that are then byte-encoded and BPE-merged independently.
        // Pattern matched (in order):
        //   's | 't | 're | 've | 'm | 'll | 'd
        //   ' ?' followed by [A-Za-z]+
        //   ' ?' followed by [0-9]+
        //   ' ?' followed by run of non-space, non-letter, non-digit
        //   whitespace runs (final-only or other)
        // Non-ASCII bytes are passed through as 1-byte pieces.
        std::vector<std::string> pre_tokenize(const std::string& text) const
        {
            std::vector<std::string> out;
            const std::size_t n = text.size();
            std::size_t i = 0;
            auto is_letter = [](unsigned char c) {
                return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
            };
            auto is_digit = [](unsigned char c) {
                return c >= '0' && c <= '9';
            };
            auto is_space = [](unsigned char c) {
                return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f';
            };
            while (i < n) {
                // Contractions: 's, 't, 're, 've, 'm, 'll, 'd
                if (text[i] == '\'') {
                    static const char* const contractions[] = {"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"};
                    bool matched = false;
                    for (const char* c : contractions) {
                        const std::size_t L = std::strlen(c);
                        if (i + L <= n && text.compare(i, L, c) == 0) {
                            out.emplace_back(text.substr(i, L));
                            i += L;
                            matched = true;
                            break;
                        }
                    }
                    if (matched) continue;
                }

                // GPT-2 attaches an optional leading space to letters/digits/
                // punctuation pieces. Non-space whitespace (\t, \n, ...) is
                // handled by the whitespace-run branch.
                const bool leading_space = (text[i] == ' ');
                const std::size_t start = i;
                const std::size_t look = leading_space ? i + 1 : i;

                if (look < n && is_letter(static_cast<unsigned char>(text[look]))) {
                    std::size_t j = look;
                    while (j < n && is_letter(static_cast<unsigned char>(text[j]))) ++j;
                    out.emplace_back(text.substr(start, j - start));
                    i = j;
                    continue;
                }
                if (look < n && is_digit(static_cast<unsigned char>(text[look]))) {
                    std::size_t j = look;
                    while (j < n && is_digit(static_cast<unsigned char>(text[j]))) ++j;
                    out.emplace_back(text.substr(start, j - start));
                    i = j;
                    continue;
                }
                if (look < n
                    && !is_letter(static_cast<unsigned char>(text[look]))
                    && !is_digit(static_cast<unsigned char>(text[look]))
                    && !is_space(static_cast<unsigned char>(text[look]))) {
                    std::size_t j = look;
                    while (j < n
                        && !is_letter(static_cast<unsigned char>(text[j]))
                        && !is_digit(static_cast<unsigned char>(text[j]))
                        && !is_space(static_cast<unsigned char>(text[j]))) {
                        ++j;
                    }
                    out.emplace_back(text.substr(start, j - start));
                    i = j;
                    continue;
                }
                // No content match — text[look] is whitespace or out of range.
                // text[i] must be whitespace (since the only way ``look``
                // skips a character is leading_space).
                if (is_space(static_cast<unsigned char>(text[i]))) {
                    std::size_t j = i;
                    while (j < n && is_space(static_cast<unsigned char>(text[j]))) ++j;
                    const std::size_t run = j - i;
                    // Emit the whole run except for a single trailing space
                    // before non-space content; that space attaches to the
                    // next piece via the ``leading_space`` logic above.
                    if (j < n && text[j - 1] == ' ' && run >= 2) {
                        out.emplace_back(text.substr(i, run - 1));
                        i = j - 1;
                    } else {
                        out.emplace_back(text.substr(i, run));
                        i = j;
                    }
                    continue;
                }
                // Fallback: emit one byte at a time (handles non-ASCII).
                out.emplace_back(text.substr(i, 1));
                ++i;
            }
            return out;
        }

        // Apply BPE merges to a single byte-encoded piece, appending the
        // resulting token ids to ``out``. Operates on a vector of UTF-8
        // codepoints (each codepoint is one BPE "symbol" initially).
        void bpe_encode(const std::string& piece, std::vector<int>& out) const
        {
            if (piece.empty()) return;
            // Split into codepoints.
            std::vector<std::string> symbols;
            std::size_t i = 0;
            while (i < piece.size()) {
                const unsigned char c = static_cast<unsigned char>(piece[i]);
                std::size_t len = 1;
                if ((c & 0x80) == 0) len = 1;
                else if ((c & 0xE0) == 0xC0) len = 2;
                else if ((c & 0xF0) == 0xE0) len = 3;
                else if ((c & 0xF8) == 0xF0) len = 4;
                if (i + len > piece.size()) len = piece.size() - i;
                symbols.emplace_back(piece.substr(i, len));
                i += len;
            }

            // Iteratively merge the lowest-rank adjacent pair.
            while (symbols.size() > 1) {
                int best_rank = std::numeric_limits<int>::max();
                std::size_t best_idx = symbols.size();
                for (std::size_t k = 0; k + 1 < symbols.size(); ++k) {
                    auto it = merge_ranks_.find({ symbols[k], symbols[k + 1] });
                    if (it != merge_ranks_.end() && it->second < best_rank) {
                        best_rank = it->second;
                        best_idx = k;
                    }
                }
                if (best_idx == symbols.size()) break;
                symbols[best_idx] += symbols[best_idx + 1];
                symbols.erase(symbols.begin() + static_cast<std::ptrdiff_t>(best_idx) + 1);
            }

            for (const auto& sym : symbols) {
                auto it = vocab_.find(sym);
                if (it != vocab_.end()) {
                    out.push_back(it->second);
                } else {
                    // Unknown symbol: emit each byte as its own id (must exist
                    // since the byte-level BPE always covers all 256 bytes).
                    for (char ch : sym) {
                        const unsigned char ub = static_cast<unsigned char>(ch);
                        const auto& enc = byte_to_unicode_[ub];
                        auto it2 = vocab_.find(enc);
                        if (it2 != vocab_.end()) out.push_back(it2->second);
                    }
                }
            }
        }

        struct pair_hash {
            std::size_t operator()(const std::pair<std::string, std::string>& p) const noexcept
            {
                return std::hash<std::string>()(p.first) ^ (std::hash<std::string>()(p.second) << 1);
            }
        };

        // Declaration order matters: ``unicode_to_byte_`` is filled in by
        // ``make_byte_to_unicode`` (called from the initialiser of
        // ``byte_to_unicode_``), so it must be constructed first.
        std::unordered_map<std::string, uint8_t> unicode_to_byte_;
        std::array<std::string, 256> byte_to_unicode_;
        std::unordered_map<std::string, int> vocab_;
        std::vector<std::string> inv_vocab_;
        std::unordered_map<std::pair<std::string, std::string>, int, pair_hash> merge_ranks_;
    };

}  // namespace llm
}  // namespace fdeep

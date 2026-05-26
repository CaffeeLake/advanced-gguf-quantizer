#pragma once

#include <stdexcept>
#include <string>
#include <vector>

#if !defined(_WIN32)
#include <termios.h>
#endif

namespace bq::tui {

enum class Key {
    UP,
    DOWN,
    ENTER,
    ESC,
    SLASH,
    CSI,
    OTHER,
};

struct PromptReadResult {
    std::string value;
    bool cancelled = false;
    bool eof = false;
};

class PromptCancelled : public std::runtime_error {
public:
    PromptCancelled();
};

class RawTerminal {
public:
    RawTerminal();
    ~RawTerminal();

    RawTerminal(const RawTerminal &) = delete;
    RawTerminal & operator=(const RawTerminal &) = delete;

private:
#if !defined(_WIN32)
    bool enabled = false;
    termios old_term{};
#endif
};

bool stdin_is_tty();
bool read_input_char(char & c, int timeout_ms);
bool consume_escape_sequence_after_esc(Key & key, int timeout_ms = 40);
Key read_key();

std::string strip_control_input(const std::string & input);
PromptReadResult read_prompt_line();
int menu_select(const std::string & title, const std::vector<std::string> & options);

} // namespace bq::tui

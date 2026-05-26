#include "terminal_ui.h"
#include "tui.h"

#include <cctype>
#include <cerrno>
#include <iostream>

#if !defined(_WIN32)
#include <sys/select.h>
#include <sys/types.h>
#include <unistd.h>
#endif

namespace bq::tui {

#if !defined(_WIN32)
static bool read_stdin_char(char & c, int timeout_ms) {
    fd_set set;
    FD_ZERO(&set);
    FD_SET(STDIN_FILENO, &set);

    timeval tv{};
    timeval * tv_ptr = nullptr;
    if (timeout_ms >= 0) {
        tv.tv_sec = timeout_ms / 1000;
        tv.tv_usec = (timeout_ms % 1000) * 1000;
        tv_ptr = &tv;
    }

    int ready = 0;
    do {
        ready = select(STDIN_FILENO + 1, &set, nullptr, nullptr, tv_ptr);
    } while (ready < 0 && errno == EINTR);
    if (ready <= 0) {
        return false;
    }

    ssize_t n = 0;
    do {
        n = read(STDIN_FILENO, &c, 1);
    } while (n < 0 && errno == EINTR);
    return n == 1;
}
#endif

static int pending_input_char = -1;

PromptCancelled::PromptCancelled() : std::runtime_error("input cancelled") {}

static std::vector<MenuOption> menu_options_from_labels(const std::vector<std::string> & labels) {
    std::vector<MenuOption> options;
    options.reserve(labels.size());
    for (const std::string & label : labels) {
        options.push_back({ label, "", "" });
    }
    return options;
}

RawTerminal::RawTerminal() {
#if !defined(_WIN32)
    enabled = stdin_is_tty();
    if (!enabled || tcgetattr(STDIN_FILENO, &old_term) != 0) {
        enabled = false;
        return;
    }
    termios raw = old_term;
    raw.c_lflag &= ~(ICANON | ECHO | IEXTEN);
    raw.c_iflag &= ~(IXON | ICRNL);
    raw.c_cc[VMIN] = 1;
    raw.c_cc[VTIME] = 0;
    if (tcsetattr(STDIN_FILENO, TCSANOW, &raw) != 0) {
        enabled = false;
    }
#endif
}

RawTerminal::~RawTerminal() {
#if !defined(_WIN32)
    if (enabled) {
        tcsetattr(STDIN_FILENO, TCSANOW, &old_term);
    }
#endif
}

bool stdin_is_tty() {
#if !defined(_WIN32)
    return isatty(STDIN_FILENO) == 1;
#else
    return false;
#endif
}

bool read_input_char(char & c, int timeout_ms) {
    if (pending_input_char >= 0) {
        c = (char) pending_input_char;
        pending_input_char = -1;
        return true;
    }
#if !defined(_WIN32)
    return read_stdin_char(c, timeout_ms);
#else
    (void) timeout_ms;
    return (bool) std::cin.get(c);
#endif
}

static void unread_input_char(char c) {
    pending_input_char = (unsigned char) c;
}

bool consume_escape_sequence_after_esc(Key & key, int timeout_ms) {
    key = Key::ESC;
#if !defined(_WIN32)
    if (!stdin_is_tty()) {
        return false;
    }

    char first = 0;
    if (!read_input_char(first, timeout_ms)) {
        return false;
    }
    if (first != '[' && first != 'O') {
        unread_input_char(first);
        return false;
    }

    char c = 0;
    for (int i = 0; i < 16; ++i) {
        if (!read_input_char(c, timeout_ms)) {
            key = Key::CSI;
            return true;
        }
        if (c >= '@' && c <= '~') {
            if (c == 'A') {
                key = Key::UP;
            } else if (c == 'B') {
                key = Key::DOWN;
            } else {
                key = Key::CSI;
            }
            return true;
        }
    }
    key = Key::CSI;
    return true;
#else
    (void) timeout_ms;
    return false;
#endif
}

Key read_key() {
    char c = 0;
#if !defined(_WIN32)
    if (stdin_is_tty()) {
        if (!read_input_char(c, -1)) {
            return Key::ESC;
        }
    } else
#endif
    {
    if (!std::cin.get(c)) {
        return Key::ESC;
    }
    }
    if (c == '\n' || c == '\r') {
        return Key::ENTER;
    }
    if (c == 27) {
        char b1 = 0;
        char b2 = 0;
#if !defined(_WIN32)
        if (stdin_is_tty()) {
            Key escaped = Key::ESC;
            (void) consume_escape_sequence_after_esc(escaped);
            return escaped;
        } else
#endif
        {
        if (!std::cin.get(b1) || (b1 != '[' && b1 != 'O') || !std::cin.get(b2)) {
            return Key::ESC;
        }
        }
        if (b2 == 'A') {
            return Key::UP;
        }
        if (b2 == 'B') {
            return Key::DOWN;
        }
        return Key::OTHER;
    }
    if (c == 'k' || c == 'K') {
        return Key::UP;
    }
    if (c == 'j' || c == 'J') {
        return Key::DOWN;
    }
    if (c == '/') {
        return Key::SLASH;
    }
    return Key::OTHER;
}

std::string strip_control_input(const std::string & input) {
    std::string out;
    out.reserve(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        const unsigned char c = (unsigned char) input[i];
        if (c == 27 && i + 1 < input.size()) {
            if (input[i + 1] == '[' || input[i + 1] == 'O') {
                i += 2;
                while (i < input.size() && (input[i] < '@' || input[i] > '~')) {
                    ++i;
                }
                continue;
            }
            continue;
        }
        if (c >= 32 || c == '\t') {
            out.push_back((char) c);
        }
    }
    return out;
}

PromptReadResult read_prompt_line() {
    PromptReadResult result;
#if !defined(_WIN32)
    if (stdin_is_tty()) {
        RawTerminal raw;
        while (true) {
            char c = 0;
            if (!read_input_char(c, -1)) {
                result.eof = true;
                std::cout << "\n";
                return result;
            }
            const unsigned char uc = (unsigned char) c;
            if (c == '\r' || c == '\n') {
                std::cout << "\n";
                return result;
            }
            if (c == 4) {
                result.cancelled = true;
                result.eof = true;
                std::cout << "\n";
                return result;
            }
            if (c == 27) {
                Key escaped = Key::ESC;
                if (!consume_escape_sequence_after_esc(escaped)) {
                    result.cancelled = true;
                    std::cout << "\n";
                    return result;
                }
                if (escaped == Key::ESC) {
                    result.cancelled = true;
                    std::cout << "\n";
                    return result;
                }
                continue;
            }
            if (c == 21) {
                while (!result.value.empty()) {
                    result.value.pop_back();
                    std::cout << "\b \b";
                }
                std::cout << std::flush;
                continue;
            }
            if (c == 127 || c == 8) {
                if (!result.value.empty()) {
                    result.value.pop_back();
                    std::cout << "\b \b" << std::flush;
                }
                continue;
            }
            if (std::isprint(uc) || c == '\t') {
                result.value.push_back(c);
                std::cout << c << std::flush;
            }
        }
    }
#endif
    if (!std::getline(std::cin, result.value)) {
        result.eof = true;
    }
    result.value = strip_control_input(result.value);
    return result;
}

int menu_select(const std::string & title, const std::vector<std::string> & options) {
    if (options.empty()) {
        throw std::runtime_error("empty menu: " + title);
    }

#if !defined(_WIN32)
    if (stdin_is_tty()) {
        RawTerminal raw;
        int selected = 0;
        const std::vector<MenuOption> menu_options = menu_options_from_labels(options);
        while (true) {
            const TerminalCapabilities caps = detect_terminal(stdout);
            if (caps.ansi) {
                std::cout << "\033[2J\033[H";
            }
            print(std::cout, render_menu(title, menu_options, (size_t) selected, caps));
            std::cout << paint("Use Up/Down or j/k, Enter to select, Esc to go back.", muted(), caps) << "\n";
            std::cout << std::flush;
            switch (read_key()) {
                case Key::UP:
                    selected = (selected + (int) options.size() - 1) % (int) options.size();
                    break;
                case Key::DOWN:
                    selected = (selected + 1) % (int) options.size();
                    break;
                case Key::ENTER:
                    std::cout << "\033[2J\033[H";
                    return selected;
                case Key::ESC:
                    std::cout << "\033[2J\033[H";
                    return (int) options.size() - 1;
                case Key::SLASH:
                    break;
                default:
                    break;
            }
        }
    }
#endif

    const TerminalCapabilities caps = detect_terminal(stdout);
    print(std::cout, render_menu(title, menu_options_from_labels(options), 0, caps));
    std::cout << "choice: " << std::flush;
    std::string line;
    std::getline(std::cin, line);
    line = strip_control_input(line);
    if (line.empty() || line == "q" || line == "quit" || line == "back" || line == "cancel") {
        return (int) options.size() - 1;
    }
    const int choice = std::stoi(line);
    if (choice < 1 || choice > (int) options.size()) {
        throw std::runtime_error("menu choice out of range");
    }
    return choice - 1;
}

} // namespace bq::tui

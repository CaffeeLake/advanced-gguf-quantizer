#include "tui.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <sstream>

#if !defined(_WIN32)
#include <sys/ioctl.h>
#include <unistd.h>
#endif

namespace bq::tui {
namespace {

static int color_code(Color color, bool background) {
    const int base = background ? 40 : 30;
    switch (color) {
        case Color::Black:         return base + 0;
        case Color::Red:           return base + 1;
        case Color::Green:         return base + 2;
        case Color::Yellow:        return base + 3;
        case Color::Blue:          return base + 4;
        case Color::Magenta:       return base + 5;
        case Color::Cyan:          return base + 6;
        case Color::White:         return base + 7;
        case Color::BrightBlack:   return (background ? 100 : 90) + 0;
        case Color::BrightRed:     return (background ? 100 : 90) + 1;
        case Color::BrightGreen:   return (background ? 100 : 90) + 2;
        case Color::BrightYellow:  return (background ? 100 : 90) + 3;
        case Color::BrightBlue:    return (background ? 100 : 90) + 4;
        case Color::BrightMagenta: return (background ? 100 : 90) + 5;
        case Color::BrightCyan:    return (background ? 100 : 90) + 6;
        case Color::BrightWhite:   return (background ? 100 : 90) + 7;
        case Color::Default:       return background ? 49 : 39;
    }
    return background ? 49 : 39;
}

static std::string repeat(char c, size_t count) {
    return std::string(count, c);
}

static std::string pad_right(const std::string & text, size_t width) {
    const size_t shown = display_width(text);
    if (shown >= width) {
        return text;
    }
    return text + repeat(' ', width - shown);
}

static std::string pad_left(const std::string & text, size_t width) {
    const size_t shown = display_width(text);
    if (shown >= width) {
        return text;
    }
    return repeat(' ', width - shown) + text;
}

static size_t layout_width(const TerminalCapabilities & caps) {
    if (caps.is_tty && caps.columns > 12) {
        return std::min<size_t>((size_t) caps.columns - 4, 112);
    }
    return 96;
}

static size_t menu_visible_rows(const TerminalCapabilities & caps, size_t option_count) {
    if (!caps.is_tty || caps.rows <= 0) {
        return option_count;
    }
    const size_t rows = (size_t) caps.rows;
    const size_t available = rows > 21 ? rows - 21 : 3;
    return std::max<size_t>(3, std::min<size_t>(option_count, std::min<size_t>(available, 18)));
}

static size_t clamp_frame_width(size_t requested, const TerminalCapabilities & caps) {
    const size_t max_width = layout_width(caps);
    if (requested == 0) {
        requested = max_width;
    }
    return std::max<size_t>(12, std::min(requested, max_width));
}

static std::string left_label(const std::string & label, const TerminalCapabilities & caps) {
    return paint(label, muted(), caps);
}

static std::string getenv_or_empty(const char * name) {
    const char * value = std::getenv(name);
    return value == nullptr ? std::string() : std::string(value);
}

static std::string trim_copy(const std::string & text) {
    size_t begin = 0;
    while (begin < text.size() && std::isspace((unsigned char) text[begin])) {
        ++begin;
    }

    size_t end = text.size();
    while (end > begin && std::isspace((unsigned char) text[end - 1])) {
        --end;
    }
    return text.substr(begin, end - begin);
}

static bool punctuation_only(const std::string & text) {
    const std::string trimmed = trim_copy(text);
    if (trimmed.empty()) {
        return true;
    }
    return std::all_of(trimmed.begin(), trimmed.end(), [](unsigned char c) {
        return c == ':' || c == '-' || c == '>';
    });
}

static std::string prompt_label_or_default(const std::string & label) {
    std::string trimmed = trim_copy(label);
    while (!trimmed.empty() && trimmed.back() == ':') {
        trimmed.pop_back();
    }
    trimmed = trim_copy(trimmed);
    if (punctuation_only(trimmed)) {
        return "Input";
    }
    if (trimmed == "Project name") {
        return "Project Name";
    }
    return trimmed;
}

static std::string display_menu_label(const std::string & label) {
    if (label == "Create new project") {
        return "Create New Project";
    }
    if (label == "Load existing project") {
        return "Load Existing Project";
    }
    if (label == "Select model input/output") {
        return "Select Model Input/Output";
    }
    if (label == "Select model input/output now") {
        return "Select Model Input/Output Now";
    }
    if (label == "Select options") {
        return "Select Options";
    }
    if (label == "Start quantization") {
        return "Start Quantization";
    }
    if (label == "Show status") {
        return "Show Status";
    }
    if (label == "Model input and output") {
        return "Model Input/Output";
    }
    if (label == "Quality inputs") {
        return "Quality Inputs";
    }
    if (label == "Set target BPW / VRAM") {
        return "Set Target BPW / VRAM";
    }
    if (label == "NVFP4 4/6 Policy") {
        return "NVFP4 4/6 Policy";
    }
    if (label == "Native candidate search") {
        return "Native Candidate Search";
    }
    if (label == "Select native technique families") {
        return "Select Native Technique Families";
    }
    if (label == "Use full local GGUF search set") {
        return "Use Full Local GGUF Search Set";
    }
    if (label == "Mixed NVFP4/MXFP6 policy") {
        return "Mixed NVFP4/MXFP6 Policy";
    }
    if (label == "MXFP6 scale refinement") {
        return "MXFP6 Scale Refinement";
    }
    if (label == "Tensor rules") {
        return "Tensor Rules";
    }
    if (label == "Standard quantize options") {
        return "Standard Quantize Options";
    }
    if (label == "Save config") {
        return "Save Config";
    }
    if (label == "Use this directory") {
        return "Use This Directory";
    }
    if (label == "Type directory path") {
        return "Type Directory Path";
    }
    if (label == "Type path") {
        return "Type Path";
    }
    if (label == "Filter names") {
        return "Filter Names";
    }
    if (label == "Filter directories") {
        return "Filter Directories";
    }
    if (label == "Clear filter") {
        return "Clear Filter";
    }
    if (label == "Previous page") {
        return "Previous Page";
    }
    if (label == "Next page") {
        return "Next Page";
    }
    return label;
}

} // namespace

TerminalCapabilities detect_terminal(FILE * stream) {
    TerminalCapabilities caps;
#if !defined(_WIN32)
    const int fd = fileno(stream);
    caps.is_tty = isatty(fd) == 1;
    winsize ws{};
    if (caps.is_tty && ioctl(fd, TIOCGWINSZ, &ws) == 0) {
        if (ws.ws_col > 0) {
            caps.columns = ws.ws_col;
        }
        if (ws.ws_row > 0) {
            caps.rows = ws.ws_row;
        }
    }
#else
    (void) stream;
#endif

    const std::string term = getenv_or_empty("TERM");
    const std::string colorterm = getenv_or_empty("COLORTERM");
    caps.dumb = term == "dumb";
    caps.ansi = caps.is_tty && !caps.dumb;
    caps.truecolor = colorterm == "truecolor" || colorterm == "24bit";
    caps.color = should_use_color(caps);
    return caps;
}

bool should_use_color(const TerminalCapabilities & caps) {
    if (!caps.ansi || caps.dumb) {
        return false;
    }
    if (std::getenv("NO_COLOR") != nullptr) {
        return false;
    }
    return true;
}

Style accent()   { return { Color::BrightCyan,  Color::Default, true,  false, false }; }
Style success()  { return { Color::BrightGreen, Color::Default, true,  false, false }; }
Style warning()  { return { Color::BrightYellow, Color::Default, true, false, false }; }
Style error()    { return { Color::BrightRed,   Color::Default, true,  false, false }; }
Style muted()    { return { Color::BrightBlack, Color::Default, false, true,  false }; }
Style selected() { return { Color::Black,       Color::BrightCyan, true, false, false }; }
Style normal()   { return { Color::BrightWhite, Color::Default, false, false, false }; }

std::string ansi(const Style & style, const TerminalCapabilities & caps) {
    if (!caps.color) {
        return {};
    }
    std::vector<int> codes;
    if (style.bold) {
        codes.push_back(1);
    }
    if (style.dim) {
        codes.push_back(2);
    }
    if (style.inverse) {
        codes.push_back(7);
    }
    if (style.fg != Color::Default) {
        codes.push_back(color_code(style.fg, false));
    }
    if (style.bg != Color::Default) {
        codes.push_back(color_code(style.bg, true));
    }
    if (codes.empty()) {
        return {};
    }

    std::ostringstream out;
    out << "\033[";
    for (size_t i = 0; i < codes.size(); ++i) {
        if (i > 0) {
            out << ';';
        }
        out << codes[i];
    }
    out << 'm';
    return out.str();
}

std::string reset(const TerminalCapabilities & caps) {
    return caps.color ? "\033[0m" : "";
}

std::string paint(std::string text, const Style & style, const TerminalCapabilities & caps) {
    if (!caps.color) {
        return text;
    }
    return ansi(style, caps) + text + reset(caps);
}

std::string strip_ansi(const std::string & text) {
    std::string out;
    out.reserve(text.size());
    for (size_t i = 0; i < text.size(); ++i) {
        if (text[i] == '\033' && i + 1 < text.size() && text[i + 1] == '[') {
            i += 2;
            while (i < text.size() && (text[i] < '@' || text[i] > '~')) {
                ++i;
            }
            continue;
        }
        out.push_back(text[i]);
    }
    return out;
}

size_t display_width(const std::string & text) {
    return strip_ansi(text).size();
}

std::vector<std::string> wrap_words(const std::string & text, size_t width) {
    if (width == 0 || text.size() <= width) {
        return { text };
    }

    std::vector<std::string> lines;
    std::istringstream in(text);
    std::string word;
    std::string line;
    while (in >> word) {
        if (line.empty()) {
            line = word;
        } else if (display_width(line) + 1 + display_width(word) <= width) {
            line += " " + word;
        } else {
            lines.push_back(line);
            line = word;
        }

        while (display_width(line) > width) {
            lines.push_back(line.substr(0, width));
            line.erase(0, width);
        }
    }
    if (!line.empty()) {
        lines.push_back(line);
    }
    if (lines.empty()) {
        lines.push_back("");
    }
    return lines;
}

std::string truncate_to_width(const std::string & text, size_t width) {
    if (display_width(text) <= width) {
        return text;
    }
    if (width <= 3) {
        return text.substr(0, width);
    }
    return strip_ansi(text).substr(0, width - 3) + "...";
}

std::string horizontal_rule(size_t width, char fill) {
    return repeat(fill, width);
}

std::string render_box(const std::vector<std::string> & lines, const BoxOptions & options, const TerminalCapabilities & caps) {
    const size_t padding = std::max(0, options.padding);
    size_t content_width = options.width > 0 && (size_t) options.width > padding * 2 + 2
        ? (size_t) options.width - padding * 2 - 2
        : 0;
    for (const std::string & line : lines) {
        content_width = std::max(content_width, display_width(line));
    }
    if (!options.title.empty()) {
        content_width = std::max(content_width, display_width(options.title) + 2);
    }

    const size_t width = clamp_frame_width(content_width + padding * 2 + 2, caps);
    const size_t text_width = width > padding * 2 + 2 ? width - padding * 2 - 2 : 1;
    const std::string border = repeat('-', width - 2);

    std::ostringstream out;
    out << paint("+" + border + "+", options.border_style, caps) << "\n";
    if (!options.title.empty()) {
        const std::string title = " " + truncate_to_width(options.title, width > 6 ? width - 6 : width) + " ";
        const size_t left = (width - 2 > display_width(title)) ? (width - 2 - display_width(title)) / 2 : 0;
        const size_t right = width - 2 > display_width(title) + left ? width - 2 - display_width(title) - left : 0;
        out << paint("|", options.border_style, caps)
            << repeat(' ', left)
            << paint(title, options.title_style, caps)
            << repeat(' ', right)
            << paint("|", options.border_style, caps)
            << "\n";
        out << paint("+" + border + "+", options.border_style, caps) << "\n";
    }

    for (const std::string & raw : lines) {
        const std::vector<std::string> wrapped = options.wrap ? wrap_words(raw, text_width) : std::vector<std::string>{ truncate_to_width(raw, text_width) };
        for (const std::string & line : wrapped) {
            out << paint("|", options.border_style, caps)
                << repeat(' ', padding)
                << pad_right(line, text_width)
                << repeat(' ', padding)
                << paint("|", options.border_style, caps)
                << "\n";
        }
    }
    out << paint("+" + border + "+", options.border_style, caps) << "\n";
    return out.str();
}

std::string render_section_header(
        const std::string & title,
        const std::string & subtitle,
        const TerminalCapabilities & caps,
        size_t width) {
    size_t resolved_width = width > 0 ? width : layout_width(caps);
    resolved_width = std::min(resolved_width, layout_width(caps));
    resolved_width = std::max<size_t>(resolved_width, 12);

    std::ostringstream out;
    if (!title.empty()) {
        out << paint(title, accent(), caps) << "\n";
    }
    if (!subtitle.empty()) {
        const size_t text_width = std::min<size_t>(resolved_width, 96);
        for (const std::string & line : wrap_words(subtitle, text_width)) {
            out << paint(line, muted(), caps) << "\n";
        }
    }
    if (!title.empty() || !subtitle.empty()) {
        out << paint(repeat('-', std::min<size_t>(resolved_width, 80)), muted(), caps) << "\n";
    }
    return out.str();
}

std::string render_input_prompt(const InputPrompt & prompt, const TerminalCapabilities & caps) {
    const size_t width = layout_width(caps);
    const std::string label = prompt_label_or_default(prompt.label);
    const std::string placeholder = punctuation_only(prompt.placeholder) ? "type a value" : trim_copy(prompt.placeholder);

    std::ostringstream out;
    out << paint(label, accent(), caps);
    if (prompt.required) {
        out << " " << paint("required", muted(), caps);
    }
    out << "\n";

    std::ostringstream current;
    current << "  "
            << left_label("Current", caps) << "  "
            << paint(prompt.value.empty() ? "-" : prompt.value,
                    prompt.value.empty() ? muted() : prompt.value_style,
                    caps);
    out << truncate_to_width(current.str(), width) << "\n";

    if (!prompt.hint.empty()) {
        const size_t hint_width = width > 2 ? width - 2 : width;
        for (const std::string & line : wrap_words(prompt.hint, hint_width)) {
            out << "  " << paint(line, muted(), caps) << "\n";
        }
    }

    if (!placeholder.empty()) {
        out << "  " << paint(truncate_to_width(placeholder, width > 2 ? width - 2 : width), muted(), caps) << "\n";
    }

    out << paint(prompt.value.empty() ? "Value > " : "New value > ", accent(), caps);
    return out.str();
}

std::string render_branded_header(const ProductInfo & product, const TerminalCapabilities & caps) {
    std::ostringstream title;
    title << paint(product.name, accent(), caps)
          << " "
          << paint("v" + product.version, muted(), caps);
    const size_t width = layout_width(caps);
    std::ostringstream out;
    out << title.str() << "\n";
    if (!product.subtitle.empty()) {
        out << paint(product.subtitle, muted(), caps) << "\n";
    }
    out << paint(repeat('-', std::min<size_t>(width, 80)), muted(), caps) << "\n";
    return out.str();
}

std::string render_status_panel(
        const ProductInfo & product,
        const std::vector<StatusItem> & items,
        const TerminalCapabilities & caps) {
    (void) product;
    const size_t width = layout_width(caps);
    size_t label_width = 0;
    for (const StatusItem & item : items) {
        label_width = std::max(label_width, display_width(item.label));
    }
    label_width = std::min<size_t>(label_width, 16);

    std::vector<std::string> rows;
    rows.reserve(items.size());
    for (const StatusItem & item : items) {
        std::ostringstream row;
        row << left_label(pad_right(item.label, label_width), caps) << "  "
            << paint(item.value.empty() ? "-" : item.value, item.value_style, caps);
        if (!item.detail.empty()) {
            row << " " << paint(item.detail, muted(), caps);
        }
        rows.push_back(truncate_to_width(row.str(), width > 6 ? width - 6 : width));
    }
    BoxOptions box;
    box.title = "Session";
    box.width = (int) width;
    box.padding = 1;
    box.wrap = false;
    box.border_style = { Color::BrightBlue, Color::Default, false, false, false };
    box.title_style = accent();
    return render_box(rows, box, caps) + "\n";
}

std::vector<SlashCommand> default_slash_commands() {
    return {
        { "/help", "", "show shell commands" },
        { "/commands", "", "show shell commands" },
        { "/status", "", "show project, model, output, quant type, and run status" },
        { "/project", "[path]", "open an existing project" },
        { "/config", "[path]", "load a saved configuration" },
        { "/model", "[path]", "set the active BF16/input GGUF path" },
        { "/output", "[path]", "set the active output GGUF path" },
        { "/candidates", "[dir]", "generate NVFP4 4/6 and mixed candidate configs" },
        { "/metrics", "", "record PPL/KLD/BPW metrics for a project variant" },
        { "/best", "[metrics]", "run real PPL/KLD best-candidate evaluation" },
        { "/inspect", "[path]", "inspect a GGUF, defaulting to active model/output" },
        { "/policy", "", "explain stock MOSTLY_x higher-bitrate choices" },
        { "/run", "", "run active configuration when ready" },
        { "/clear", "", "redraw the shell" },
        { "/quit", "", "leave the shell" },
    };
}

std::string render_slash_help(const std::vector<SlashCommand> & commands, const TerminalCapabilities & caps) {
    size_t name_width = 0;
    for (const SlashCommand & command : commands) {
        name_width = std::max(name_width, display_width(command.name + (command.args.empty() ? "" : " " + command.args)));
    }
    name_width = std::min<size_t>(name_width, 28);

    const size_t width = layout_width(caps);
    std::ostringstream out;
    out << render_section_header("Commands", "", caps, std::min<size_t>(width, 72));
    for (const SlashCommand & command : commands) {
        const std::string lhs = command.name + (command.args.empty() ? "" : " " + command.args);
        std::ostringstream row;
        row << paint(pad_right(lhs, name_width), accent(), caps)
            << "  "
            << command.description;
        out << truncate_to_width(row.str(), width) << "\n";
    }
    return out.str();
}

std::string format_menu_option(
        size_t index,
        const MenuOption & option,
        bool is_selected,
        const TerminalCapabilities & caps,
        size_t width) {
    std::ostringstream prefix;
    prefix << pad_left(std::to_string(index + 1), 2) << ". ";
    std::string label = display_menu_label(option.label);
    if (option.disabled) {
        label = paint(label, muted(), caps);
    } else if (is_selected) {
        label = paint(" " + label + " ", selected(), caps);
    } else {
        label = paint(label, normal(), caps);
    }

    std::ostringstream out;
    out << (is_selected ? paint(">", accent(), caps) : " ")
        << " "
        << paint(prefix.str(), muted(), caps)
        << label;
    const bool show_description = !option.description.empty() && option.description.size() <= 36 && width >= 96;
    if (show_description) {
        out << "  " << paint(option.description, muted(), caps);
    }
    const std::string rendered = out.str();
    if (width > 0) {
        std::string line = truncate_to_width(rendered, width);
        if (!option.description.empty() && !show_description && width >= 72) {
            const size_t detail_width = width > 8 ? width - 8 : width;
            line += "\n       " + paint(truncate_to_width(option.description, detail_width), muted(), caps);
        }
        return line;
    }
    return rendered;
}

std::string render_menu(
        const std::string & title,
        const std::vector<MenuOption> & options,
        size_t selected_index,
        const TerminalCapabilities & caps,
        size_t max_visible_rows) {
    size_t width = std::min<size_t>(layout_width(caps), 88);
    for (size_t i = 0; i < options.size(); ++i) {
        const std::string label = display_menu_label(options[i].label);
        width = std::max(width, std::min<size_t>(display_width(label) + 8, layout_width(caps)));
    }

    std::vector<std::string> rows;
    const size_t content_rows = max_visible_rows > 0 ?
        std::max<size_t>(3, std::min<size_t>(max_visible_rows, options.size())) :
        menu_visible_rows(caps, options.size());
    const size_t option_rows = content_rows < options.size() && content_rows > 2 ? content_rows - 2 : content_rows;
    rows.reserve(std::min(options.size(), content_rows) + 2);
    const size_t row_width = std::min<size_t>(width > 4 ? width - 4 : width, 68);
    size_t begin = 0;
    size_t end = options.size();
    if (option_rows < options.size()) {
        const size_t half = option_rows / 2;
        begin = selected_index > half ? selected_index - half : 0;
        end = std::min(options.size(), begin + option_rows);
        begin = end >= option_rows ? end - option_rows : 0;
    }
    if (begin > 0) {
        rows.push_back(paint("... " + std::to_string(begin) + " above", muted(), caps));
    }
    for (size_t i = begin; i < end; ++i) {
        rows.push_back(format_menu_option(i, options[i], i == selected_index, caps, row_width));
    }
    if (end < options.size()) {
        rows.push_back(paint("... " + std::to_string(options.size() - end) + " below", muted(), caps));
    }

    BoxOptions box;
    box.title = title;
    box.width = (int) width;
    box.padding = 1;
    box.wrap = false;
    box.border_style = accent();
    return render_box(rows, box, caps) + "\n";
}

void print(std::ostream & out, const std::string & rendered) {
    out << rendered;
    if (rendered.empty() || rendered.back() != '\n') {
        out << "\n";
    }
}

} // namespace bq::tui

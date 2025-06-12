#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <unordered_map>
#include <algorithm>
#include <thread>
#include <mutex>
#include <atomic>
#include <memory>
#include <iomanip>
#include <filesystem>
#include <cassert>
#include <sstream>
#include <random>

// Enable chrono literals for easier duration specification
using namespace std::chrono_literals;

// Prevent Windows min/max macro issues (only define if not already defined)
#ifndef NOMINMAX
#define NOMINMAX
#endif

// For memory-mapped file
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

/**
 * @brief Structure representing a single trade tick from Binance data format
 * Format: aggTradeId,price,qty,firstTradeId,lastTradeId,timestamp,maker
 */
struct Tick {
    std::chrono::system_clock::time_point timestamp;
    double price;
    double quantity;
    std::string symbol = "BTC";  // Default symbol since not in CSV
    int64_t aggTradeId = 0;
    int64_t firstTradeId = 0;
    int64_t lastTradeId = 0;
    bool maker = false;
};

/**
 * @brief Structure representing an OHLCV candle bar
 */
struct CandleBar {
    std::chrono::system_clock::time_point open_time;
    std::chrono::system_clock::time_point close_time;
    double open;
    double high;
    double low;
    double close;
    double volume;
    std::string symbol;
    int tick_count;
    
    // Initialize with first tick
    CandleBar(const Tick& tick) 
        : open_time(tick.timestamp), close_time(tick.timestamp),
          open(tick.price), high(tick.price), low(tick.price), close(tick.price),
          volume(tick.quantity), symbol(tick.symbol), tick_count(1) {}
    
    // Update candle with a new tick
    void update(const Tick& tick) {
        high = std::max(high, tick.price);
        low = std::min(low, tick.price);
        close = tick.price;
        volume += tick.quantity;
        close_time = tick.timestamp;
        tick_count++;
    }

    // Verify candle integrity
    bool validate() const {
        return (high >= low) && 
               (high >= open) && (high >= close) &&
               (low <= open) && (low <= close) &&
               (tick_count > 0) && (volume >= 0);
    }

    // Helper for debugging/testing
    std::string toString() const {
        std::stringstream ss;
        ss << "OHLC: " << open << "/" << high << "/" << low << "/" << close 
           << " Vol: " << volume << " Ticks: " << tick_count;
        return ss.str();
    }
};

/**
 * @brief Thread-safe candle aggregator with efficient time-based bucketing
 */
class CandleAggregator {
private:
    std::chrono::seconds bar_duration;
    std::unordered_map<std::string, std::unique_ptr<CandleBar>> current_candles;
    std::vector<CandleBar> completed_candles;
    std::mutex candle_mutex;

public:
    CandleAggregator(std::chrono::seconds duration) : bar_duration(duration) {}

    // Process a single tick
    void process_tick(const Tick& tick) {
        // Calculate candle key based on timestamp
        auto bar_start = std::chrono::duration_cast<std::chrono::seconds>(
            tick.timestamp.time_since_epoch()) / bar_duration * bar_duration;
        auto candle_key = tick.symbol + "_" + std::to_string(bar_start.count());
        
        std::lock_guard<std::mutex> lock(candle_mutex);
        
        // Update or create candle
        if (current_candles.find(candle_key) == current_candles.end()) {
            current_candles[candle_key] = std::make_unique<CandleBar>(tick);
        } else {
            current_candles[candle_key]->update(tick);
        }
    }

    // Process multiple ticks
    void process_ticks(const std::vector<Tick>& ticks) {
        for (const auto& tick : ticks) {
            process_tick(tick);
        }
    }

    // Get all completed candles
    std::vector<CandleBar> get_completed_candles() {
        std::lock_guard<std::mutex> lock(candle_mutex);
        
        // Move all candles to result vector
        std::vector<CandleBar> result;
        result.reserve(current_candles.size());
        
        for (auto& pair : current_candles) {
            result.push_back(std::move(*(pair.second)));
        }
        
        // Sort by symbol and open time
        std::sort(result.begin(), result.end(), 
                  [](const CandleBar& a, const CandleBar& b) {
                      if (a.symbol != b.symbol) return a.symbol < b.symbol;
                      return a.open_time < b.open_time;
                  });
                  
        return result;
    }

    // Clear all candles
    void clear() {
        std::lock_guard<std::mutex> lock(candle_mutex);
        current_candles.clear();
        completed_candles.clear();
    }
};

// Convert Unix timestamp (milliseconds) to time_point
inline std::chrono::system_clock::time_point unix_ms_to_timepoint(int64_t unix_ms) {
    return std::chrono::system_clock::time_point(
        std::chrono::milliseconds(unix_ms));
}

// Format timestamp for output
std::string format_timestamp(const std::chrono::system_clock::time_point& tp) {
    auto time_t = std::chrono::system_clock::to_time_t(tp);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        tp.time_since_epoch() % std::chrono::seconds(1)).count();
    
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
    ss << '.' << std::setfill('0') << std::setw(3) << ms;
    return ss.str();
}

/**
 * @brief Efficient memory-mapped file reader for large CSV files
 */
class MemoryMappedFile {
private:
    const char* data = nullptr;
    size_t size = 0;

#ifdef _WIN32
    HANDLE file_handle = INVALID_HANDLE_VALUE;
    HANDLE mapping_handle = NULL;
#else
    int fd = -1;
#endif

public:
    MemoryMappedFile(const std::string& path) {
#ifdef _WIN32
        file_handle = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL, 
                                OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
        if (file_handle == INVALID_HANDLE_VALUE) {
            throw std::runtime_error("Failed to open file: " + path);
        }

        LARGE_INTEGER file_size;
        if (!GetFileSizeEx(file_handle, &file_size)) {
            CloseHandle(file_handle);
            throw std::runtime_error("Failed to get file size: " + path);
        }
        size = file_size.QuadPart;

        mapping_handle = CreateFileMapping(file_handle, NULL, PAGE_READONLY, 0, 0, NULL);
        if (mapping_handle == NULL) {
            CloseHandle(file_handle);
            throw std::runtime_error("Failed to create file mapping: " + path);
        }

        data = static_cast<const char*>(MapViewOfFile(mapping_handle, FILE_MAP_READ, 0, 0, 0));
        if (data == nullptr) {
            CloseHandle(mapping_handle);
            CloseHandle(file_handle);
            throw std::runtime_error("Failed to map view of file: " + path);
        }
#else
        fd = open(path.c_str(), O_RDONLY);
        if (fd == -1) {
            throw std::runtime_error("Failed to open file: " + path);
        }

        struct stat sb;
        if (fstat(fd, &sb) == -1) {
            close(fd);
            throw std::runtime_error("Failed to get file size: " + path);
        }
        size = sb.st_size;

        data = static_cast<const char*>(mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0));
        if (data == MAP_FAILED) {
            close(fd);
            throw std::runtime_error("Failed to map file: " + path);
        }
#endif
    }

    ~MemoryMappedFile() {
#ifdef _WIN32
        if (data != nullptr) {
            UnmapViewOfFile(data);
        }
        if (mapping_handle != NULL) {
            CloseHandle(mapping_handle);
        }
        if (file_handle != INVALID_HANDLE_VALUE) {
            CloseHandle(file_handle);
        }
#else
        if (data != MAP_FAILED) {
            munmap(const_cast<char*>(data), size);
        }
        if (fd != -1) {
            close(fd);
        }
#endif
    }

    const char* get_data() const { return data; }
    size_t get_size() const { return size; }
};

// ASCII candlestick chart for visualization
void display_ascii_chart(const std::vector<CandleBar>& candles, int width = 80, int height = 20) {
    if (candles.empty()) {
        std::cout << "No data to display.\n";
        return;
    }
    
    // Find price range
    double min_price = candles[0].low;
    double max_price = candles[0].high;
    
    for (const auto& candle : candles) {
        min_price = std::min(min_price, candle.low);
        max_price = std::max(max_price, candle.high);
    }
    
    double price_range = max_price - min_price;
    if (price_range <= 0) price_range = 1.0; // Avoid division by zero
    
    // Create the chart
    std::cout << "\n=== BTC Candlestick Chart (" << candles.size() << " bars) ===\n\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "High: " << max_price << " | Low: " << min_price << "\n\n";
    
    // Display time scale
    std::cout << format_timestamp(candles.front().open_time) << " to " 
              << format_timestamp(candles.back().close_time) << "\n\n";
    
    // Only show the most recent candles that fit in the width
    int start_idx = std::max(0, static_cast<int>(candles.size()) - width);
    
    // Display price scale on the left
    for (int y = 0; y < height; y++) {
        double price = max_price - (y * price_range / height);
        std::cout << std::setw(10) << price << " |";
        
        for (int x = 0; x < std::min(width, static_cast<int>(candles.size() - start_idx)); x++) {
            const auto& candle = candles[start_idx + x];
            
            // Map candle to chart coordinates
            int candle_top = static_cast<int>((max_price - candle.high) * height / price_range);
            int candle_bottom = static_cast<int>((max_price - candle.low) * height / price_range);
            int candle_open = static_cast<int>((max_price - candle.open) * height / price_range);
            int candle_close = static_cast<int>((max_price - candle.close) * height / price_range);
            
            // Draw the candle
            if (y >= candle_top && y <= candle_bottom) {
                if (y == candle_top || y == candle_bottom) {
                    std::cout << "|"; // Wick
                } else if (y >= std::min(candle_open, candle_close) && 
                           y <= std::max(candle_open, candle_close)) {
                    // Body of candle
                    if (candle.close >= candle.open) {
                        std::cout << "█"; // Bullish (green) candle
                    } else {
                        std::cout << "▓"; // Bearish (red) candle
                    }
                } else {
                    std::cout << "|"; // Wick
                }
            } else {
                std::cout << " ";
            }
        }
        
        std::cout << "\n";
    }
    
    // Draw time axis
    std::cout << std::string(11, ' ') << "+" << std::string(std::min(width, static_cast<int>(candles.size() - start_idx)), '-') << "\n";
}

/**
 * @brief Highly optimized parallel CSV parser using memory-mapped files
 */
std::vector<CandleBar> parallel_aggregate_ticks(const std::string& csv_path, 
                                              std::chrono::seconds bar_duration,
                                              int num_threads = 0,
                                              bool show_progress = false) {
    // Use hardware concurrency if not specified
    if (num_threads <= 0) {
        num_threads = std::thread::hardware_concurrency();
        if (num_threads == 0) num_threads = 4; // Fallback
    }
    
    // Memory map the file
    MemoryMappedFile mmfile(csv_path);
    const char* file_data = mmfile.get_data();
    const size_t file_size = mmfile.get_size();
    
    // Find end of header line
    size_t header_end = 0;
    while (header_end < file_size && file_data[header_end] != '\n') {
        header_end++;
    }
    if (header_end < file_size) header_end++; // Skip the newline
    
    // Find file chunks for parallel processing
    std::vector<std::pair<size_t, size_t>> chunks;
    size_t chunk_size = (file_size - header_end) / num_threads;
    
    for (int i = 0; i < num_threads; i++) {
        size_t start = (i == 0) ? header_end : chunks.back().second;
        size_t end = (i == num_threads - 1) ? file_size : start + chunk_size;
        
        // Adjust chunk end to next newline
        if (i < num_threads - 1) {
            while (end < file_size && file_data[end] != '\n') {
                end++;
            }
            if (end < file_size) end++; // Skip the newline
        }
        
        chunks.push_back({start, end});
    }
    
    // Create aggregator
    CandleAggregator aggregator(bar_duration);
    
    // Progress tracking
    std::atomic<size_t> processed_bytes(0);
    
    // Process in parallel
    std::vector<std::thread> threads;
    for (const auto& chunk : chunks) {
        threads.emplace_back([&aggregator, &file_data, chunk, &processed_bytes]() {
            size_t pos = chunk.first;
            
            while (pos < chunk.second) {
                // Find line boundaries
                size_t line_start = pos;
                while (pos < chunk.second && file_data[pos] != '\n') {
                    pos++;
                }
                
                // Process the line
                if (pos > line_start) {
                    // Skip empty lines
                    if (pos - line_start <= 1) {
                        pos++;
                        continue;
                    }
                    
                    // Parse CSV fields (aggTradeId,price,qty,firstTradeId,lastTradeId,timestamp,maker)
                    size_t field_start = line_start;
                    size_t field_end = line_start;
                    int field_index = 0;
                    
                    Tick tick;
                    
                    while (field_end < pos) {
                        // Find field end (comma or end of line)
                        field_end = field_start;
                        while (field_end < pos && file_data[field_end] != ',') {
                            field_end++;
                        }
                        
                        // Extract field value
                        size_t field_len = field_end - field_start;
                        std::string field_value(file_data + field_start, field_len);
                        
                        switch(field_index) {
                            case 0: // aggTradeId
                                tick.aggTradeId = std::stoll(field_value);
                                break;
                            case 1: // price
                                tick.price = std::stod(field_value);
                                break;
                            case 2: // qty (quantity)
                                tick.quantity = std::stod(field_value);
                                break;
                            case 3: // firstTradeId
                                tick.firstTradeId = std::stoll(field_value);
                                break;
                            case 4: // lastTradeId
                                tick.lastTradeId = std::stoll(field_value);
                                break;
                            case 5: // timestamp (Unix milliseconds)
                                tick.timestamp = unix_ms_to_timepoint(std::stoll(field_value));
                                break;
                            case 6: // maker (True/False)
                                tick.maker = (field_value == "True");
                                break;
                        }
                        
                        // Move to next field
                        field_index++;
                        field_start = field_end + 1;
                        
                        // Skip comma
                        if (field_end < pos) {
                            field_end++;
                        }
                    }
                    
                    // Process the tick if we got all fields
                    if (field_index >= 7) {
                        aggregator.process_tick(tick);
                    }
                }
                
                // Move past newline
                if (pos < chunk.second) {
                    pos++;
                }
                
                // Update progress
                processed_bytes += (pos - line_start);
            }
        });
    }
    
    // Show progress if requested
    if (show_progress) {
        const int bar_width = 70;
        while (processed_bytes < file_size - header_end) {
            float progress = static_cast<float>(processed_bytes) / (file_size - header_end);
            int bar_position = static_cast<int>(bar_width * progress);
            
            std::cout << "\r[";
            for (int i = 0; i < bar_width; ++i) {
                if (i < bar_position) std::cout << "=";
                else if (i == bar_position) std::cout << ">";
                else std::cout << " ";
            }
            
            std::cout << "] " << static_cast<int>(progress * 100.0) << "%"
                      << " (" << processed_bytes << "/" << (file_size - header_end) << " bytes)" 
                      << std::flush;
            
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        
        std::cout << "\r[";
        for (int i = 0; i < bar_width; ++i) {
            std::cout << "=";
        }
        std::cout << "] 100%" << std::endl;
    }
    
    // Wait for threads to complete
    for (auto& thread : threads) {
        thread.join();
    }
    
    // Get results
    return aggregator.get_completed_candles();
}

// Save candles to CSV
void save_candles_to_csv(const std::vector<CandleBar>& candles, const std::string& output_path) {
    std::ofstream out(output_path);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open output file: " + output_path);
    }
    
    // Write header
    out << "open_time,close_time,open,high,low,close,volume,tick_count\n";
    
    // Write data
    for (const auto& candle : candles) {
        out << format_timestamp(candle.open_time) << ","
            << format_timestamp(candle.close_time) << ","
            << candle.open << ","
            << candle.high << ","
            << candle.low << ","
            << candle.close << ","
            << candle.volume << ","
            << candle.tick_count << "\n";
    }
}

/**
 * @brief Generate random tick data for testing
 */
std::vector<Tick> generate_random_ticks(int count, 
                                     int64_t start_time_ms = 1717561800000,
                                     double price_mean = 70000.0,
                                     double price_volatility = 100.0,
                                     double quantity_mean = 0.1,
                                     double quantity_volatility = 0.2) {
    std::vector<Tick> ticks;
    ticks.reserve(count);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> price_dist(price_mean, price_volatility);
    std::normal_distribution<> qty_dist(quantity_mean, quantity_volatility);
    
    // Timespan for ticks (distribute over 10 minutes)
    std::uniform_int_distribution<int64_t> time_dist(0, 600000); // 10 minutes in ms
    
    int64_t trade_id_base = 3625141958;
    
    for (int i = 0; i < count; i++) {
        Tick tick;
        tick.timestamp = unix_ms_to_timepoint(start_time_ms + time_dist(gen));
        tick.price = std::max(0.01, price_dist(gen)); // Ensure positive price
        tick.quantity = std::max(0.00001, qty_dist(gen)); // Ensure positive quantity
        tick.aggTradeId = trade_id_base + i;
        tick.firstTradeId = tick.aggTradeId;
        tick.lastTradeId = tick.aggTradeId;
        tick.maker = (i % 2 == 0); // Alternate maker/taker
        
        ticks.push_back(tick);
    }
    
    // Sort by timestamp
    std::sort(ticks.begin(), ticks.end(), 
              [](const Tick& a, const Tick& b) { return a.timestamp < b.timestamp; });
              
    return ticks;
}

/**
 * @brief Custom testing with user-provided parameters
 */
void run_custom_test(int tick_count, int candle_duration_seconds, 
                    double price_mean = 70000.0, double price_volatility = 100.0) {
    std::cout << "\n=== CUSTOM TEST ===\n";
    std::cout << "Generating " << tick_count << " ticks with " 
              << candle_duration_seconds << "-second candles..." << std::endl;
    std::cout << "Price mean: " << price_mean << ", volatility: " << price_volatility << std::endl;
              
    auto ticks = generate_random_ticks(tick_count, 1717561800000, price_mean, price_volatility);
    
    // Process ticks - REPLACE THIS LINE:
    // CandleAggregator agg(std::chrono::seconds(candle_duration_seconds));
    
    // WITH THIS:
    auto agg = CandleAggregator(std::chrono::seconds(candle_duration_seconds));
    
    // Rest of your function remains unchanged
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Process all ticks
    for (const auto& tick : ticks) {
        agg.process_tick(tick);
    }
    
    auto candles = agg.get_completed_candles();
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    // Verify candles
    std::cout << "Generated " << candles.size() << " candles in "
              << duration.count() / 1000.0 << " ms" << std::endl;
    std::cout << "Processing rate: " << (tick_count * 1000.0) / duration.count() 
              << " K ticks/sec" << std::endl;
              
    // Calculate statistics
    int total_ticks = 0;
    double total_volume = 0.0;
    bool all_valid = true;
    
    for (const auto& candle : candles) {
        total_ticks += candle.tick_count;
        total_volume += candle.volume;
        
        if (!candle.validate()) {
            std::cout << "INVALID CANDLE: " << candle.toString() << std::endl;
            all_valid = false;
        }
    }
    
    std::cout << "Total processed ticks: " << total_ticks << std::endl;
    std::cout << "Total volume: " << total_volume << std::endl;
    
    if (all_valid) {
        std::cout << "All candles passed validation checks" << std::endl;
    } else {
        std::cout << "WARNING: Some candles failed validation" << std::endl;
    }
    
    // Sample candles
    if (candles.size() > 0) {
        std::cout << "\nFirst candle: " << format_timestamp(candles.front().open_time) 
                  << " to " << format_timestamp(candles.front().close_time)
                  << " " << candles.front().toString() << std::endl;
                  
        if (candles.size() > 1) {
            std::cout << "Last candle: " << format_timestamp(candles.back().open_time)
                      << " to " << format_timestamp(candles.back().close_time)
                      << " " << candles.back().toString() << std::endl;
        }
    }

    // Optional: Save test results to CSV
    std::string test_output = "test_candles.csv";
    save_candles_to_csv(candles, test_output);
    std::cout << "Test candles saved to " << test_output << std::endl;
}

/**
 * @brief Comprehensive benchmark function with detailed performance metrics
 */
void benchmark(const std::string& csv_path, const std::vector<int>& durations, 
               const std::vector<int>& thread_counts = {1, 2, 4, 8}) {
    std::cout << "\n=== BENCHMARK RESULTS ===\n";
    std::cout << "File: " << csv_path << "\n";
    double file_size_mb = std::filesystem::file_size(csv_path) / (1024.0 * 1024.0);
    std::cout << "Size: " << file_size_mb << " MB\n\n";

    std::cout << "| Duration | Threads | Time (ms) | Throughput (MB/s) | Candles | Ticks/s    |\n";
    std::cout << "|----------|---------|-----------|-----------------|---------|------------|\n";

    for (int seconds : durations) {
        for (int threads : thread_counts) {
            auto start_time = std::chrono::high_resolution_clock::now();
            
            auto candles = parallel_aggregate_ticks(csv_path, std::chrono::seconds(seconds), threads, false);
            
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            
            double throughput_mb = file_size_mb / (duration_ms.count() / 1000.0);
            
            // Calculate total ticks
            int total_ticks = 0;
            for (const auto& candle : candles) {
                total_ticks += candle.tick_count;
            }
            
            double ticks_per_second = total_ticks / (duration_ms.count() / 1000.0);
            
            std::cout << "| " << std::setw(8) << seconds << "s | " 
                      << std::setw(7) << threads << " | "
                      << std::setw(9) << duration_ms.count() << " | "
                      << std::setw(16) << std::fixed << std::setprecision(2) << throughput_mb << " | "
                      << std::setw(7) << candles.size() << " | "
                      << std::setw(10) << std::fixed << std::setprecision(0) << ticks_per_second << " |\n";
        }
        std::cout << "\n";
    }
    
    // Memory usage report
    size_t tick_size = sizeof(Tick);
    size_t candle_size = sizeof(CandleBar);
    
    std::cout << "Memory footprint:\n";
    std::cout << "- Tick struct: " << tick_size << " bytes\n";
    std::cout << "- Candle struct: " << candle_size << " bytes\n";
    std::cout << "- Memory-mapped file: Zero copy, minimal memory overhead\n";
}

int main(int argc, char* argv[]) {
    std::string csv_path = "binance_ticks.csv";
    std::string output_path = "candles.csv";
    int bar_duration = 13;  // Default: 13-second bars
    int num_threads = 0;    // Auto-detect
    bool show_chart = false; // Default: don't show chart
    bool run_custom_testing = false;
    int custom_test_ticks = 10000;
    double custom_price_mean = 70000.0;
    double custom_price_vol = 100.0;
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-f" || arg == "--file") {
            if (i + 1 < argc) csv_path = argv[++i];
        } else if (arg == "-o" || arg == "--output") {
            if (i + 1 < argc) output_path = argv[++i];
        } else if (arg == "-d" || arg == "--duration") {
            if (i + 1 < argc) bar_duration = std::stoi(argv[++i]);
        } else if (arg == "-t" || arg == "--threads") {
            if (i + 1 < argc) num_threads = std::stoi(argv[++i]);
        } else if (arg == "-c" || arg == "--chart") {
            show_chart = true;
        } else if (arg == "-b" || arg == "--benchmark") {
            benchmark(csv_path, {1, 5, 13, 30, 60, 300});
            return 0;
        } else if (arg == "--test") {
            run_custom_testing = true;
            // Parse additional test parameters if provided
            if (i + 1 < argc && argv[i+1][0] != '-') {
                custom_test_ticks = std::stoi(argv[++i]);
                if (i + 1 < argc && argv[i+1][0] != '-') {
                    custom_price_mean = std::stod(argv[++i]);
                    if (i + 1 < argc && argv[i+1][0] != '-') {
                        custom_price_vol = std::stod(argv[++i]);
                    }
                }
            }
        } else if (arg == "-h" || arg == "--help") {
            std::cout << "Tick Data Aggregator - High Performance C++ CSV Processor\n\n"
                      << "Usage: " << argv[0] << " [options]\n"
                      << "Options:\n"
                      << "  -f, --file <path>       Input tick CSV file (default: tick_data.csv)\n"
                      << "  -o, --output <path>     Output candle CSV file (default: candles.csv)\n"
                      << "  -d, --duration <secs>   Candle duration in seconds (default: 13)\n"
                      << "  -t, --threads <num>     Number of threads (default: auto)\n"
                      << "  -c, --chart             Display ASCII candlestick chart\n"
                      << "  -b, --benchmark         Run comprehensive benchmarks\n"
                      << "  --test [count] [price] [vol]  Run custom test with parameters:\n"
                      << "                          count: number of ticks (default: 10000)\n"
                      << "                          price: base price (default: 70000.0)\n"
                      << "                          vol: price volatility (default: 100.0)\n"
                      << "  -h, --help              Show this help message\n\n"
                      << "Example: " << argv[0] << " -f trades.csv -o 13sec_candles.csv -d 13 -c\n";
            return 0;
        }
    }
    
    // Run custom test if requested
    if (run_custom_testing) {
        run_custom_test(custom_test_ticks, bar_duration, custom_price_mean, custom_price_vol);
        return 0;
    }
    
    try {
        // Report settings
        std::cout << "=== Tick Data Aggregator ===\n";
        std::cout << "Input file:     " << csv_path << "\n";
        std::cout << "Output file:    " << output_path << "\n";
        std::cout << "Bar duration:   " << bar_duration << " seconds\n";
        std::cout << "Threads:        " << (num_threads ? std::to_string(num_threads) : "auto") << "\n";
        if (show_chart) std::cout << "Chart:          Enabled\n";
        std::cout << "\nProcessing...\n";
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Process data
        auto candles = parallel_aggregate_ticks(csv_path, std::chrono::seconds(bar_duration), 
                                              num_threads, true);
        save_candles_to_csv(candles, output_path);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        // Report results
        std::cout << "\nCompleted in " << duration.count() << " ms\n";
        std::cout << "Created " << candles.size() << " candle bars\n";
        std::cout << "Results saved to " << output_path << "\n";
        
        double throughput = (std::filesystem::file_size(csv_path) / (1024.0 * 1024.0)) / 
                           (duration.count() / 1000.0);
        std::cout << "Throughput: " << std::fixed << std::setprecision(2) << throughput << " MB/s\n";
        
        // Display chart if requested
        if (show_chart) {
            display_ascii_chart(candles, 80, 20);
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
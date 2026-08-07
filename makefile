CC = gcc
CFLAGS  = -O3 -march=native -ffast-math -Iinclude -fopenmp -std=c17 \
			-fopt-info-all=opt_details.txt
LDFLAGS = -lm

SRC_DIR = src
TEST_DIR = tests
BENCH_DIR = benchmarks
BIN_DIR = bin

CORE_SRC = \
	$(SRC_DIR)/prng.c \
	$(SRC_DIR)/vector.c \
	$(SRC_DIR)/matrix.c \
	$(SRC_DIR)/logging.c \
	$(SRC_DIR)/config.c \
	$(SRC_DIR)/fourier.c

CORE_OBJ = $(CORE_SRC:.c=.o)

MAIN_SRC = $(SRC_DIR)/main.c
MAIN_OBJ = $(MAIN_SRC:.c=.o)
TARGET = $(BIN_DIR)/main


# ---------------------------------------------------------
# Default target
# ---------------------------------------------------------
all: $(TARGET)

# ---------------------------------------------------------
# Ensure bin directory exists
# ---------------------------------------------------------
$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(TARGET): $(BIN_DIR) $(CORE_OBJ) $(MAIN_OBJ)
	$(CC) $(CFLAGS) $(CORE_OBJ) $(MAIN_OBJ) -o $@ $(LDFLAGS)

# Compile core and main objects
src/%.o: src/%.c
	$(CC) $(CFLAGS) -c $< -o $@

# ---------------------------------------------------------
# TEST BUILDING SECTION
# ---------------------------------------------------------
TESTS = $(basename $(notdir $(wildcard $(TEST_DIR)/*.c)))
TEST_BIN = $(addprefix $(BIN_DIR)/, $(TESTS))

.PHONY: tests
tests: $(TEST_BIN)

$(BIN_DIR)/test_%: $(TEST_DIR)/test_%.c $(CORE_OBJ) | $(BIN_DIR)
	$(CC) $(CFLAGS) $(CORE_OBJ) $< -o $@ $(LDFLAGS)


# ---------------------------------------------------------
# OPTIMIZATION REPORT
# ---------------------------------------------------------

OPT_REPORT = opt_report.txt
.PHONY: opt_report
opt_report: $(OPT_REPORT)
	@echo "======================================"> $(OPT_REPORT)
	@echo "    OPTIMIZATION REPORT" >> $(OPT_REPORT)
	@echo "======================================">> $(OPT_REPORT)

	@echo "Compiler:" >> $(OPT_REPORT)
	@echo "  CC      = $(CC)" >> $(OPT_REPORT)
	@echo "  Version = $$( $(CC) --version | head -n 1 )" >> $(OPT_REPORT)
	@echo "" >> $(OPT_REPORT)

	@echo "Flags:" >> $(OPT_REPORT)
	@echo "  CFLAGS  = $(CFLAGS)" >> $(OPT_REPORT)
	@echo "  LDFLAGS = $(LDFLAGS)" >> $(OPT_REPORT)
	@echo "" >> $(OPT_REPORT)
	
	@echo "Object File Sizes:" >> $(OPT_REPORT)
	@for f in $(CORE_OBJ); do \
        if [ -f $$f ]; then \
            printf "  %-30s %10s\n" "$$f" "$$(stat -c%s $$f) bytes" >> $(OPT_REPORT); \
        fi; \
	done
	@echo "" >> $(OPT_REPORT)

	@echo "Benchmark Binary Sizes:" >> $(OPT_REPORT)
	@for b in $(BENCH_BIN); do \
	    if [ -f $$b ]; then \
	        printf "  %-30s %10s\n" "$$b" "$$(stat -c%s $$b) bytes" >> $(OPT_REPORT); \
	    fi; \
	done
	@echo "" >> $(OPT_REPORT)

	@echo "Build Timestamp:" >> $(OPT_REPORT)
	@date >> $(OPT_REPORT)

	@echo "" >> $(OPT_REPORT)
	@echo "Report written to $(OPT_REPORT)"
	    @echo "" >> $(OPT_REPORT)

	@echo "======================================" >> $(OPT_REPORT)
	@echo "    FULL GCC OPTIMIZATION REPORT" >> $(OPT_REPORT)
	@echo "======================================" >> $(OPT_REPORT)
	@echo "" >> $(OPT_REPORT)
	
	@if [ -f opt_details.txt ]; then \
	    echo "Appending GCC optimization details..." >> $(OPT_REPORT); \
	    cat opt_details.txt >> $(OPT_REPORT); \
	else \
	    echo "No GCC optimization details found (opt_details.txt missing)" >> $(OPT_REPORT); \
	fi


# ---------------------------------------------------------
# BENCHMARK BUILDING SECTION
# ---------------------------------------------------------
BENCHMARKS = $(basename $(notdir $(wildcard $(BENCH_DIR)/*.c)))
BENCH_BIN = $(addprefix $(BIN_DIR)/, $(BENCHMARKS))

.PHONY: benchmarks
benchmarks: $(BENCH_BIN) opt_report

$(BIN_DIR)/bench_%: $(BENCH_DIR)/bench_%.c $(CORE_OBJ) | $(BIN_DIR)
	$(CC) $(CFLAGS) $(CORE_OBJ) $< -o $@ $(LDFLAGS)

# ---------------------------------------------------------
# CLEAN
# ---------------------------------------------------------
clean:
	rm -f $(CORE_OBJ) $(MAIN_OBJ)
	rm -rf $(BIN_DIR)


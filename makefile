CC = gcc
CFLAGS  = -O3 -march=native -ffast-math -Iinclude -fopenmp -std=c17
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
# BENCHMARK BUILDING SECTION
# ---------------------------------------------------------
BENCHMARKS = $(basename $(notdir $(wildcard $(BENCH_DIR)/*.c)))
BENCH_BIN = $(addprefix $(BIN_DIR)/, $(BENCHMARKS))

.PHONY: benchmarks
benchmarks: $(BENCH_BIN)

$(BIN_DIR)/bench_%: $(BENCH_DIR)/bench_%.c $(CORE_OBJ) | $(BIN_DIR)
	$(CC) $(CFLAGS) $(CORE_OBJ) $< -o $@ $(LDFLAGS)

# ---------------------------------------------------------
# CLEAN
# ---------------------------------------------------------
clean:
	rm -f $(CORE_OBJ) $(MAIN_OBJ)
	rm -rf $(BIN_DIR)


CC = gcc
CFLAGS  = -O3 -march=native -ffast-math -Iinclude
LDFLAGS = -lm

SRC_DIR = src
TEST_DIR = tests

CORE_SRC = \
	$(SRC_DIR)/prng.c \
	$(SRC_DIR)/vector.c \
	$(SRC_DIR)/matrix.c \
	$(SRC_DIR)/logging.c

CORE_OBJ = $(CORE_SRC:.c=.o)

MAIN_SRC = $(SRC_DIR)/main.c
MAIN_OBJ = $(MAIN_SRC:.c=.o)
TARGET = main

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) $(CORE_OBJ) $(MAIN_OBJ) -o $(TARGET) $(LDFLAGS)

# Compile core and main objects
src/%.o: src/%.c
	$(CC) $(CFLAGS) -c $< -o $@


# ---------------------------------------------------------
# TEST BUILDING SECTION
# ---------------------------------------------------------
# Pattern rule:
# "make test_matrix" builds test/test_matrix.c + all SRC objects
# Output binary: test_matrix
# ---------------------------------------------------------
TESTS = $(basename $(notdir $(wildcard $(TEST_DIR)/*.c)))

$(TESTS): %: $(TEST_DIR)/%.c $(OBJ)
	$(CC) $(CFLAGS) $(CORE_OBJ) $< -o $@ $(LDFLAGS)

clean:
	rm -f $(CORE_OBJ) $(MAIN_OBJ) $(TARGET) $(TESTS)


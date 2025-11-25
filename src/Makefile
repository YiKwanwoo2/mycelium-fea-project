# -----------------------------
# User Configurable Paths
# -----------------------------
PETSC_DIR  ?= /home/kwyi/petsc
PETSC_ARCH ?= arch-linux-c-opt

CXX = mpicxx
CXXFLAGS   = -std=c++17 -O3 -Wall -Wextra -Wpedantic
INC        = -I$(PETSC_DIR)/include -I$(PETSC_DIR)/$(PETSC_ARCH)/include
LIBS = -L$(PETSC_DIR)/$(PETSC_ARCH)/lib -lpetsc -Wl,-rpath,$(PETSC_DIR)/$(PETSC_ARCH)/lib
FS_LIB     = -lstdc++fs


# Output executable
TARGET     = fea_petsc_solverAndPC.exe

# Source file
SRC        = fea_petsc_solverAndPC.cpp



# -----------------------------
# Build rules
# -----------------------------
all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(INC) $(SRC) $(LIBS) $(FS_LIB) -o $(TARGET)

clean:
	rm -f $(TARGET) *.o

.PHONY: all clean

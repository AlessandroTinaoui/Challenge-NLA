CXX := g++
CXXFLAGS := -std=c++17 -Wall -Wextra -Wno-missing-field-initializers
EIGEN_FLAGS := $(shell pkg-config --cflags eigen3)

TARGET := main1
SOURCE := challenge1/challenge1.cpp

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(CXX) $(CXXFLAGS) $(EIGEN_FLAGS) $(SOURCE) -o $(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all clean

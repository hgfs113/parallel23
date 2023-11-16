CC = g++
CFLAGS = -std=c++17 -Xpreprocessor -fopenmp
LDFLAGS = -lomp -L/opt/homebrew/opt/libomp/lib

SOURCES = print.cpp backend.cpp opt.cpp main.cpp
OBJECTS = $(SOURCES:.cpp=.o)
EXECUTABLE = prog

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(CFLAGS) $(OBJECTS) -o $(EXECUTABLE) $(LDFLAGS)

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(EXECUTABLE)

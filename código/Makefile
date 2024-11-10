TARGET = heat
CC = gcc
CFLAGS = -Wall -Wextra -O2

all: $(TARGET)

$(TARGET): heat.o
	$(CC) $(CFLAGS) -o $(TARGET) heat.o

heat.o: heat.c
	$(CC) $(CFLAGS) -c heat.c

clean:
	rm -f $(TARGET) heat.o
g++ snapshot_bot.cc -std=c++11 -Wall -Wpedantic -pthread `pkg-config --libs --cflags opencv` -I $GENN_ROBOTICS_PATH -o snapshot_bot
g++ computer.cc -std=c++11 -Wall -Wpedantic -pthread `pkg-config --libs --cflags opencv` -I $GENN_ROBOTICS_PATH -o computer

g++ snapshot_bot.cc -std=c++11 -pthread `pkg-config --libs --cflags opencv` -I $GENN_ROBOTICS_PATH -o snapshot_bot

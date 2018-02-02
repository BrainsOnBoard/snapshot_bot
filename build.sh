g++ snapshot_bot.cc -std=c++11 `pkg-config --libs --cflags opencv` -I $GENN_ROBOTICS_PATH/common -o snapshot_bot

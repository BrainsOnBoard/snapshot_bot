WITH_EIGEN=1
include $(BOB_ROBOTICS_PATH)/make_common/bob_robotics.mk

COMMON_SOURCES		:= memory.cc image_input.cc mb_memory.cc mb_memory_ardin.cc mb_memory_hog.cc
SNAPSHOT_BOT_SOURCES	:= snapshot_bot.cc $(COMMON_SOURCES)
SNAPSHOT_BOT_OBJECTS	:= $(SNAPSHOT_BOT_SOURCES:.cc=.o)
SNAPSHOT_BOT_DEPS	:= $(SNAPSHOT_BOT_SOURCES:.cc=.d)

OFFLINE_TRAIN_SOURCES	:= offline_train.cc $(COMMON_SOURCES)
OFFLINE_TRAIN_OBJECTS	:= $(OFFLINE_TRAIN_SOURCES:.cc=.o)
OFFLINE_TRAIN_DEPS	:= $(OFFLINE_TRAIN_SOURCES:.cc=.d)

BENCHMARK_SOURCES	:= benchmark.cc $(COMMON_SOURCES)
BENCHMARK_OBJECTS	:= $(BENCHMARK_SOURCES:.cc=.o)
BENCHMARK_DEPS		:= $(BENCHMARK_SOURCES:.cc=.d)

VALIDATE_SOURCES	:= validate.cc $(COMMON_SOURCES)
VALIDATE_OBJECTS	:= $(VALIDATE_SOURCES:.cc=.o)
VALIDATE_DEPS		:= $(VALIDATE_SOURCES:.cc=.d)

LINK_FLAGS		+= -ldl

.PHONY: all clean

all: snapshot_bot computer offline_train benchmark validate

snapshot_bot: $(SNAPSHOT_BOT_OBJECTS)
	$(CXX) -o $@ $(SNAPSHOT_BOT_OBJECTS) $(CXXFLAGS) $(LINK_FLAGS)

offline_train: $(OFFLINE_TRAIN_OBJECTS)
	$(CXX) -o $@ $(OFFLINE_TRAIN_OBJECTS) $(CXXFLAGS) $(LINK_FLAGS)

benchmark: $(BENCHMARK_OBJECTS)
	$(CXX) -o $@ $(BENCHMARK_OBJECTS) $(CXXFLAGS) $(LINK_FLAGS)

validate: $(VALIDATE_OBJECTS)
	$(CXX) -o $@ $(VALIDATE_OBJECTS) $(CXXFLAGS) $(LINK_FLAGS)


-include $(SNAPSHOT_BOT_DEPS)
-include $(OFFLINE_TRAIN_DEPS)
-include $(BENCHMARK_DEPS)
-include $(VALIDATE_DEPS)

%.o: %.cc %.d
	$(CXX) -c -o $@ $< $(CXXFLAGS)

-include computer.d

computer: computer.cc computer.d
	$(CXX) -o $@ $< $(CXXFLAGS) $(LINK_FLAGS)

%.d: ;

clean:
	rm -f offline_train computer snapshot_bot benchmark validate *.d *.o

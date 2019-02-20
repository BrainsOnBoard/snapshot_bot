WITH_EIGEN=1
include $(BOB_ROBOTICS_PATH)/make_common/bob_robotics.mk

SNAPSHOT_BOT_SOURCES	:= snapshot_bot.cc memory.cc image_input.cc
SNAPSHOT_BOT_OBJECTS	:= $(SNAPSHOT_BOT_SOURCES:.cc=.o)
SNAPSHOT_BOT_DEPS	:= $(SNAPSHOT_BOT_SOURCES:.cc=.d)

OFFLINE_TRAIN_SOURCES	:= offline_train.cc memory.cc image_input.cc
OFFLINE_TRAIN_OBJECTS	:= $(OFFLINE_TRAIN_SOURCES:.cc=.o)
OFFLINE_TRAIN_DEPS	:= $(OFFLINE_TRAIN_SOURCES:.cc=.d)


.PHONY: all clean

all: snapshot_bot computer offline_train

snapshot_bot: $(SNAPSHOT_BOT_OBJECTS)
	$(CXX) -o $@ $(SNAPSHOT_BOT_OBJECTS) $(CXXFLAGS) $(LINK_FLAGS)

offline_train: $(OFFLINE_TRAIN_OBJECTS)
	$(CXX) -o $@ $(OFFLINE_TRAIN_OBJECTS) $(CXXFLAGS) $(LINK_FLAGS)

-include $(OFFLINE_TRAIN_DEPS)
-include $(SNAPSHOT_BOT_DEPS)

%.o: %.cc %.d
	$(CXX) -c -o $@ $< $(CXXFLAGS)

-include computer.d

computer: computer.cc computer.d
	$(CXX) -o $@ $< $(CXXFLAGS) $(LINK_FLAGS)

%.d: ;

clean:
	rm -f offline_train computer snapshot_bot *.d *.o

include $(BOB_ROBOTICS_PATH)/make_common/bob_robotics.mk

.PHONY: all clean

all: snapshot_bot computer

-include snapshot_bot.d

snapshot_bot: snapshot_bot.cc snapshot_bot.d
	$(CXX) -o $@ $< $(CXXFLAGS) $(LINK_FLAGS)

-include computer.d

computer: computer.cc computer.d
	$(CXX) -o $@ $< $(CXXFLAGS) $(LINK_FLAGS)

%.d: ;

clean:
	rm -f snapshot_bot *.d
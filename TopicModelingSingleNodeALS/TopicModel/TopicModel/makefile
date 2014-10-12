SRC=TopicModel.cpp IOfn.cpp Probability.cpp Pvalue.cpp Spectral.cpp stdafx.cpp Util.cpp
CFLAGS=-lsvd -lm
COPTZ=-O3 -g  -I.
SUPP_WARN=-Wno-write-strings

exe-topicmodel: TopicModel.cpp IOfn.cpp Probability.cpp Pvalue.cpp Spectral.cpp stdafx.cpp Util.cpp # IOfn.h Probability.h Pvalue.h Spectral.h stdafx.h Util.h
	g++ -o $@ $^ $(COPTZ) $(SUPP_WARN) -DNDEBUG -std=c++0x

run:
	./exe-topicmodel

clean:
	rm exe-topicmodel

SRC=TopicModel.cpp IOfn.cpp Probability.cpp Pvalue.cpp Spectral.cpp stdafx.cpp Util.cpp
CFLAGS=-lsvd -lm
COPTZ=-O3 -g  -I.
SUPP_WARN=-Wno-write-strings

exe-topicmodel: TopicModel.cpp IOfn.cpp Probability.cpp Pvalue.cpp Spectral.cpp stdafx.cpp Util.cpp # IOfn.h Probability.h Pvalue.h Spectral.h stdafx.h Util.h
	g++ -o $@ $^ $(COPTZ) $(SUPP_WARN) -DNDEBUG -std=c++0x

run:
	./exe-topicmodel 5352 5352 27648  5  0.1 1 ../datasets/BMC/samples_train.txt ../datasets/BMC/samples_train.txt ../datasets/BMC/result/alpha.txt ../datasets/BMC/result/beta.txt ../datasets/BMC/result/hi.txt

clean:
	rm exe-topicmodel

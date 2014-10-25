TopicModeling
=============


citation:

@article{DBLP:journals/corr/HuangNHVA13,
author    = {Furong Huang and
Niranjan U. N and
Mohammad Umar Hakeem and
Prateek Verma and
Animashree Anandkumar},
title     = {Fast Detection of Overlapping Communities via Online Tensor Methods
on GPUs},
journal   = {CoRR},
year      = {2013},
volume    = {abs/1309.0787},
url       = {http://arxiv.org/abs/1309.0787},
timestamp = {Sat, 25 Oct 2014 03:19:58 +0200},
biburl    = {http://dblp.uni-trier.de/rec/bib/journals/corr/HuangNHVA13},
bibsource = {dblp computer science bibliography, http://dblp.org}
}



================
Single node topic model learning and inference via method of moments using tensor decomposition. 
Alternating least squares with pre-processing (a whitening step consists of orthogonalization and dimensionality reduction) is implemented. 

Synthetic Data Generator: 
		TopicModeling/SyntheticDataGenerator.m

Data folder is: 
		$(SolutionDir)\datasets\

Input Arguments:

//=========================================================================
	// User Manual: 
	// (1) Data specs
	InputArgument 1: NX is the training sample size
	InputArgument 2: NX_test is the test sample size
	InputArgument 3: NA is the vocabulary size
	InputArgument 4: KHID is the number of topics you want to learn
	InputArgument 5: alpha0 is the mixing parameter, usually set to < 1
	InputArgument 6: DATATYPE denotes the index convention. 
	// -> DATATYPE == 1 assumes MATLAB index which starts from 1,DATATYPE ==0 assumes C++ index which starts from 0 .
	// e.g.  10000 100 500 3 0.01 1 
	const char* FILE_GA = argv[7];
	const char* FILE_GA_test = argv[8];
	// (2) Input files
	InputArgument 7: $(SolutionDir)\datasets\$(CorpusName)\samples_train.txt 
	InputArgument 8: $(SolutionDir)\datasets\$(CorpusName)\samples_test.txt 
	// e.g. $(SolutionDir)datasets\synthetic\samples_train.txt $(SolutionDir)datasets\synthetic\samples_test.txt
	const char* FILE_alpha_WRITE = argv[9];
	const char* FILE_beta_WRITE = argv[10];
	const char* FILE_hi_WRITE = argv[11];
	// (3) Output files
	InputArgument 9: FILE_alpha_WRITE denotes the filename for estimated topic marginal distribution
	InputArgument 10: FILE_beta_WRITE denotes the filename for estimated topic-word probability matrix
	InputArgument 11: FILE_hi_WRITE denote the estimation of topics per document for the test data. 
	// The format is:
	// $(SolutionDir)\datasets\$(CorpusName)\result\alpha.txt 
	// $(SolutionDir)\datasets\$(CorpusName)\result\beta.txt 	
	// $(SolutionDir)\datasets\$(CorpusName)\result\hi.txt 
	// e.g. $(SolutionDir)datasets\synthetic\result\alpha.txt $(SolutionDir)datasets\synthetic\result\beta.txt $(SolutionDir)datasets\synthetic\result\hi.txt
	//=====================================================================


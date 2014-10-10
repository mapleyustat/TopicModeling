//
//  main
//  TopicModel.cpp
//
//  Created by Furong Huang on 9/25/13.
//  Copyright (c) 2013 Furong Huang. All rights reserved.
//

#include "stdafx.h"
#define _CRT_SECURE_NO_WARNINGS
using namespace Eigen;
using namespace std;
clock_t TIME_start, TIME_end;
int NX;
int NX_test;
int NA;
int KHID;
double alpha0;
int DATATYPE;
int main(int argc, const char * argv[])
{
	
	NX = furong_atoi(argv[1]);
	NX_test = furong_atoi(argv[2]);
	NA = furong_atoi(argv[3]);
	KHID = furong_atoi(argv[4]);
	alpha0 = furong_atof(argv[5]);
	DATATYPE = furong_atoi(argv[6]);
	//===============================================================================================================================================================
	// User Manual: 
	// (1) Data specs
	// NX is the training sample size
	// NX_test is the test sample size
	// NA is the vocabulary size
	// KHID is the number of topics you want to learn
	// alpha0 is the mixing parameter, usually set to < 1
	// DATATYPE denotes the index convention. 
	// -> DATATYPE == 1 assumes MATLAB index which starts from 1,DATATYPE ==0 assumes C++ index which starts from 0 .
	// e.g.  10000 100 500 3 0.01 1 
	const char* FILE_GA = argv[7];
	const char* FILE_GA_test = argv[8];
	// (2) Input files
	// $(SolutionDir)\datasets\$(CorpusName)\samples_train.txt 
	// $(SolutionDir)\datasets\$(CorpusName)\samples_test.txt 
	// e.g. $(SolutionDir)datasets\synthetic\samples_train.txt $(SolutionDir)datasets\synthetic\samples_test.txt
	const char* FILE_alpha_WRITE = argv[9];
	const char* FILE_beta_WRITE = argv[10];
	const char* FILE_hi_WRITE = argv[11];
	// (3) Output files
	// FILE_alpha_WRITE denotes the filename for estimated topic marginal distribution
	// FILE_beta_WRITE denotes the filename for estimated topic-word probability matrix
	// FILE_hi_WRITE denote the estimation of topics per document for the test data. 
	// The format is:
	// $(SolutionDir)\datasets\$(CorpusName)\result\alpha.txt 
	// $(SolutionDir)\datasets\$(CorpusName)\result\beta.txt 	
	// $(SolutionDir)\datasets\$(CorpusName)\result\hi.txt 
	// e.g. $(SolutionDir)datasets\synthetic\result\alpha.txt $(SolutionDir)datasets\synthetic\result\beta.txt $(SolutionDir)datasets\synthetic\result\hi.txt
	//==============================================================================================================================================================
	TIME_start = clock();
	SparseMatrix<double> Gx_a(NX, NA);	Gx_a.resize(NX, NA);	
	Gx_a.makeCompressed();
	Gx_a = read_G_sparse((char *)FILE_GA, "Word Counts Training Data", NX, NA);
	TIME_end = clock();
	double time_readfile = double(TIME_end - TIME_start) / CLOCKS_PER_SEC;
	printf("Exec Time reading matrices before preproc = %5.10e (Seconds)\n", time_readfile);



	cout << "(1) Whitening--------------------------" << endl;
	TIME_start = clock();
	SparseMatrix<double> W(NA, KHID); W.resize(NA, KHID); W.makeCompressed();	
	VectorXd mu_a(NA); 
	SparseMatrix<double> Uw(NA, KHID);  Uw.resize(NA, KHID); Uw.makeCompressed();
	SparseMatrix<double> diag_Lw_sqrt_inv_s(KHID, KHID); diag_Lw_sqrt_inv_s.resize(NA, KHID); diag_Lw_sqrt_inv_s.makeCompressed();
	VectorXd Lengths(NX);
	second_whiten_topic(Gx_a, W, mu_a, Uw, diag_Lw_sqrt_inv_s, Lengths);

	// whitened datapoints
	SparseMatrix<double> Data_a_G = W.transpose() * Gx_a.transpose();	VectorXd Data_a_mu = W.transpose() * mu_a;

	TIME_end = clock();
	double time_whitening = double(TIME_end - TIME_start) / CLOCKS_PER_SEC;
	printf("time taken by whitening = %5.10e (Seconds)\n", time_whitening);

	cout << "(1.5) Matricization---------------------" << endl;
	MatrixXd T = MatrixXd::Zero(KHID, KHID*KHID);
	Compute_M3_topic((MatrixXd)Data_a_G, Data_a_mu, Lengths, T);

	cout << "(2) Tensor decomposition----------------" << endl;
	TIME_start = clock();
	VectorXd lambda(KHID);
	MatrixXd phi_new(KHID, KHID);

	// tensorDecom_alpha0_topic(Data_a_G, Data_a_mu, Lengths, lambda, phi_new);
	tensorDecom_batchALS(T, lambda, phi_new);
	cout << "K space eigenvectors: " << endl << phi_new << endl;
	cout << "K space eigenvalues: " << endl << lambda << endl;

	TIME_end = clock();
	double time_stpm = double(TIME_end - TIME_start) / CLOCKS_PER_SEC;
	printf("time taken by whitening = %5.10e (Seconds)\n", time_stpm);


	cout << "(3) Unwhitening-----------" << endl;
	TIME_start = clock();
	MatrixXd Inv_Lambda = (pinv_vector(lambda)).asDiagonal();
	SparseMatrix<double> inv_lam_phi = (Inv_Lambda.transpose() * phi_new.transpose()).sparseView();
	SparseMatrix<double> pi_tmp1 = inv_lam_phi * W.transpose();
	VectorXd alpha(KHID);
	MatrixXd beta(NA, KHID);
	Unwhitening(lambda, phi_new, Uw, diag_Lw_sqrt_inv_s, alpha, beta);
	
	TIME_end = clock();
	double time_post = double(TIME_end - TIME_start) / CLOCKS_PER_SEC;
	printf("time taken for post processing = %5.10e (Seconds)\n", time_post);
	
	cout << "(4) Writing results----------" << endl;
	write_alpha((char *)FILE_alpha_WRITE, alpha);
	write_beta((char *)FILE_beta_WRITE, beta);


	// decode
	cout << "(5) Decoding-----------" << endl;
	TIME_start = clock();

	SparseMatrix<double> Gx_a_test(NX_test, NA); Gx_a_test.resize(NX_test, NA);
	Gx_a_test.makeCompressed();
	Gx_a_test = read_G_sparse((char *)FILE_GA_test, "Word Counts Test Data", NX_test, NA);
	double nx_test = (double)Gx_a_test.rows();
	double na = (double)Gx_a_test.cols();
	VectorXd OnesPseudoA = VectorXd::Ones(na);
	VectorXd OnesPseudoX = VectorXd::Ones(nx_test);
	VectorXd lengths_test = Gx_a_test * OnesPseudoA;
	lengths_test = lengths_test.cwiseMax(3.0*OnesPseudoX);
	int inference = decode(alpha, beta, lengths_test, Gx_a_test, (char*)FILE_hi_WRITE);
	TIME_end = clock();
	double time_decode = double(TIME_end - TIME_start) / CLOCKS_PER_SEC;
	printf("time taken for decoding = %5.10e (Seconds)\n", time_decode);


	cout << "(6) Program over------------" << endl;
	printf("\ntime taken for execution of the whole program = %5.10e (Seconds)\n", time_whitening + time_stpm + time_post);
	return 0;
}


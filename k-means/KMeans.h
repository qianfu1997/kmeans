#pragma once
#ifndef m_KMEANS
#define m_KMEANS
/***********************************************
**
**
**
**/
#include<iostream>
#include<vector>
#include<opencv\cxcore.hpp>
#include<opencv2\opencv.hpp>
#include"Models.h"
using namespace cv;
using namespace std;

//kmeans����㷨����
class kMeansFilters {
public:
	//�ճ�ʼ�����������ݳ�ʼ��,�������ļ�
	kMeansFilters();
	kMeansFilters(Mat dataSet,int clusterNums,Mat initialVector=Mat::ones(1,1,CV_64F)*(-1));
	kMeansFilters(string dataFile);
	//��������������ݼ�����
	//initialThetaΪָ�������̬�ֲ����������
	//Mat�Ĵ�СΪclusterNums*featureNums*2(��ֵ+���
	//���ɵ�����Ϊdouble��
	kMeansFilters(int clusterNums, int featureNums, Mat initialTheta);

	//����ѵ��,����һ��model��ķ���ֵ,maxTimesָ�����ѵ������
	Models trainClusters(int maxTimes=10000);
	

	int d_numOfExamples;
	int d_numOfFeatures;
	int d_numOfClusters;
	bool isInited;


private:
	//����Ԥ����
	Mat preProcessing();
	//���ɳ�ʼ��������
	//kinds:0-�������
	Models createInitVector();

	//��ȡ��index�۴صľ�ֵ����
	//Mat��1��featureNums��
	Mat meanOfCluster(vector<int> indexs);
	Mat getCluster(vector<int> indexs);

	Mat m_dataSet;
	Mat m_dataClass;
	Mat m_minVal;
	Mat m_maxVal;
	Models m_vectorsOfCenters;

};

#endif
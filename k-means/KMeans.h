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

//kmeans相关算法集成
class kMeansFilters {
public:
	//空初始化，样本数据初始化,样本集文件
	kMeansFilters();
	kMeansFilters(Mat dataSet,int clusterNums,Mat initialVector=Mat::ones(1,1,CV_64F)*(-1));
	kMeansFilters(string dataFile);
	//随机生成样本数据集方法
	//initialTheta为指定随机正态分布满足的条件
	//Mat的大小为clusterNums*featureNums*2(均值+方差）
	//生成的样本为double集
	kMeansFilters(int clusterNums, int featureNums, Mat initialTheta);

	//聚类训练,返回一个model类的返回值,maxTimes指定最大训练次数
	Models trainClusters(int maxTimes=10000);
	

	int d_numOfExamples;
	int d_numOfFeatures;
	int d_numOfClusters;
	bool isInited;


private:
	//数据预处理
	Mat preProcessing();
	//生成初始特征向量
	//kinds:0-随机生成
	Models createInitVector();

	//求取第index聚簇的均值中心
	//Mat（1，featureNums）
	Mat meanOfCluster(vector<int> indexs);
	Mat getCluster(vector<int> indexs);

	Mat m_dataSet;
	Mat m_dataClass;
	Mat m_minVal;
	Mat m_maxVal;
	Models m_vectorsOfCenters;

};

#endif
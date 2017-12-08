#pragma once
#ifndef m_Models
#define m_Models
#include<iostream>
#include<string>
#include<vector>
#include<fstream>
#include<opencv2\opencv.hpp>
using namespace std;
using namespace cv;

class Models {
public:
	
	Models();
	//kinds:0-int,1=doubles
	Models(int clusterNums, int featureNums, int kinds = 1);
	//从文件中读取数据
	Models(string fileName );
	//对第index的聚簇中心向量进行更新
	bool modelUpdate(int index, Mat features);
	//返回第一个int为model里的编号，第二个为欧式距离
	pair<int, double> matchModel(Mat featuresOfExample);
	Mat getModels();
	//各聚簇间的方差
	bool setThreshold(int index, Mat cluster);
	vector<double> getThreshold();
	//将模型输出到文件
	bool outputModels(string fileName = string("model.txt"));

	//总聚簇数
	int m_clusterNums;
	//特征数
	int m_featureNums;
	//矩阵类型
	int m_kind;
	//模型是否生成完毕
	bool isInited;

private:
	//double euclidDistance(Mat featuresOfExample);
	Mat m_Model;
	vector<double> m_Threshold;
	
};



#endif

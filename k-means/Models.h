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
	//���ļ��ж�ȡ����
	Models(string fileName );
	//�Ե�index�ľ۴������������и���
	bool modelUpdate(int index, Mat features);
	//���ص�һ��intΪmodel��ı�ţ��ڶ���Ϊŷʽ����
	pair<int, double> matchModel(Mat featuresOfExample);
	Mat getModels();
	//���۴ؼ�ķ���
	bool setThreshold(int index, Mat cluster);
	vector<double> getThreshold();
	//��ģ��������ļ�
	bool outputModels(string fileName = string("model.txt"));

	//�ܾ۴���
	int m_clusterNums;
	//������
	int m_featureNums;
	//��������
	int m_kind;
	//ģ���Ƿ��������
	bool isInited;

private:
	//double euclidDistance(Mat featuresOfExample);
	Mat m_Model;
	vector<double> m_Threshold;
	
};



#endif

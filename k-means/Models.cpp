#include"Models.h"
using namespace std;
using namespace cv;

Models::Models(){
	m_clusterNums = 0;
	m_featureNums = 0;
	m_kind = 1;
	m_Model = Mat::zeros(1, 1, CV_64F);
	m_Threshold = vector<double>();
	isInited = false;
}

Models::Models(int clusterNums, int featureNums, int kinds) {
	m_clusterNums = clusterNums;
	m_featureNums = featureNums;
	m_kind = kinds;
	if (m_kind == 1)
		m_Model = Mat::zeros(clusterNums, featureNums, CV_64F);
	else
		m_Model = Mat::zeros(clusterNums, featureNums, CV_64F);
	m_Threshold = vector<double>(m_clusterNums);
	isInited = true;
}

Models::Models(string fileName) {
	ifstream inputFile(fileName);
	bool isWrong = false;
	char c;
	isInited = false;
	m_clusterNums = 0;
	m_featureNums = 0;
	m_kind = 1;
	m_Model = Mat::zeros(1, 1, CV_64F);
	m_Threshold = vector<double>();
	if (inputFile.is_open()) {
		if ((c = inputFile.peek()) != '\n'&&!inputFile.eof())
			inputFile >> m_clusterNums;
		if ((c = inputFile.peek()) != '\n'&&!inputFile.eof())
			inputFile >> m_featureNums;
		if (m_clusterNums > 0 && m_featureNums > 0) {
			m_Model = Mat::zeros(m_clusterNums, m_featureNums, CV_64F);
			m_Threshold = vector<double>(m_clusterNums);
			m_kind = 1;
			//加载模型
			for (int i = 0; i < m_clusterNums; i++) {
				for (int j = 0; j < m_featureNums; j++) {
					if ((c = inputFile.peek()) != '\n'&&!inputFile.eof())
						inputFile >> m_Model.at<double>(i, j);
					else {
						isWrong = true;
						break;
					}
				}
				if (isWrong)
					break;
			}

			//加载阈值
			for (int i = 0; i < m_clusterNums; i++) {
				if ((c = inputFile.peek()) != '\n'&&!inputFile.eof())
					inputFile >> m_Threshold.at(i);
				else {
					isWrong = true;
					break;
				}
			}
			isInited = true;
			if (isWrong)
				isInited = false;
		}
		inputFile.close();
	}
}

bool Models::modelUpdate(int index, Mat features) {
	bool isUpdated = false;
	if (index >= 0 && index < m_clusterNums&&features.rows == 1) {
		features.row(0).copyTo(m_Model.row(index));
		isUpdated = true;
	}
	return isUpdated;
}
//输出当前样本最匹配的模板向量id
pair<int, double> Models::matchModel(Mat featuresOfExample) {
	int minIndex = -1;
	double minDistance = -1.0;
	double tmp = 0.0;
	pair<int, double> minPair = pair<int, double>(minIndex, minDistance);
	if (featuresOfExample.rows == 1) {
		for (int i = 0; i < m_clusterNums; i++) {
			tmp = norm(m_Model.row(i), featuresOfExample.row(0), NORM_L2);
			minDistance = (minDistance == -1.0) ? minIndex=i,tmp : ((minDistance < tmp) ? minDistance : minIndex=i,tmp);
		}
		minPair = pair<int, double>(minIndex, minDistance);
	}
	return minPair;
}

Mat Models::getModels() {
	return m_Model;
}

vector<double> Models::getThreshold() {
	return m_Threshold;
}

bool Models::setThreshold(int index, Mat cluster) {
	double threshold = 0.0;
	bool isCompleted = false;
	if (!isInited)
		return isCompleted;
	if (index >= 0 && index < m_clusterNums&&cluster.cols <= m_featureNums) {
		for (int i = 0; i < cluster.rows; i++) {
			double t= norm(cluster.row(i), m_Model.row(index), NORM_L2);
			threshold = threshold > t ? threshold : t;
		}
		
		isCompleted = true;
	}
	m_Threshold.at(index) = threshold;
	return isCompleted;
}

bool Models::outputModels(string fileName) {
	bool isCompleted = false;
	ofstream outputFile(fileName);
	if (outputFile.is_open()) {
		//先输出聚簇数，特征数两个数
		outputFile << m_clusterNums << "\t" << m_featureNums << "\t" << endl;
		//循环输出聚簇向量矩阵
		for (int i = 0; i < m_clusterNums; i++) {
			for (int j = 0; j < m_featureNums; j++) 
				outputFile << m_Model.at<double>(i, j)<<"\t";
			outputFile << endl;
		}
		//循环输出聚簇阈值
		for (int i = 0; i < m_clusterNums; i++)
			outputFile << m_Threshold.at(i) << "\t";
		isCompleted = true;
		outputFile.close();
	}
	return isCompleted;
}


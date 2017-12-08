#include"KMeans.h"
#include"Models.h"
using namespace std;
using namespace cv;

kMeansFilters::kMeansFilters() {
	d_numOfExamples = 0;
	d_numOfFeatures = 0;
	d_numOfClusters = 0;
	isInited = false;
	m_dataSet = Mat::zeros(1, 1, CV_64F);
	m_dataClass = Mat::ones(1, 1, CV_64F)*(-1.0);
	m_vectorsOfCenters = Models();
}

kMeansFilters::kMeansFilters(Mat dataSet, int clusterNums,Mat initialVector) {
	d_numOfExamples = dataSet.rows;
	d_numOfFeatures = dataSet.cols;
	d_numOfClusters = clusterNums;
	m_dataSet = Mat::zeros(1, 1, CV_64F);
	m_dataClass = Mat::ones(1, 1, CV_64F)*(-1.0);
	m_vectorsOfCenters = Models();
	isInited = false;
	if (d_numOfExamples > 0 && d_numOfFeatures > 0) {
		//输入的是有效数据集
		m_dataSet = Mat::zeros(d_numOfExamples, d_numOfFeatures, CV_64F);
		m_dataClass = Mat::ones(d_numOfExamples, 2, CV_64F)*(-1.0);
		m_vectorsOfCenters = Models(d_numOfClusters, d_numOfFeatures);
		dataSet.copyTo(m_dataSet);
		if (initialVector.cols == d_numOfFeatures&&initialVector.rows==clusterNums) {
			//输入的向量特征数符合样本集
			//是有效的特征集
			
			for (int i = 0; i < d_numOfClusters; i++)
				m_vectorsOfCenters.modelUpdate(i, initialVector.row(i));
		}
		isInited = true;
	}
}

kMeansFilters::kMeansFilters(string fileName) {
	//数据文件的写法，第一行：样本数 特征数 聚簇数
	//以下一行一个数据
	ifstream inputFile(fileName);
	bool isWrong = false;
	char c;
	d_numOfExamples = 0;
	d_numOfFeatures = 0;
	d_numOfClusters = 0;
	isInited = false;
	m_dataSet = Mat::zeros(1, 1, CV_64F);
	m_dataClass = Mat::ones(1, 1, CV_64F)*(-1.0);
	m_vectorsOfCenters = Models();

	if (inputFile.is_open()) {
		if ((c = inputFile.peek()) != '\n'&&!inputFile.eof())
			inputFile >> d_numOfExamples;
		if ((c = inputFile.peek()) != '\n'&&!inputFile.eof())
			inputFile >> d_numOfFeatures;
		if ((c = inputFile.peek()) != '\n'&&!inputFile.eof())
			inputFile >> d_numOfClusters;
		if (d_numOfExamples > 0 && d_numOfFeatures > 0 && d_numOfClusters > 0) {
			//初始化数据集和向量集
			isInited = true;
			m_dataSet = Mat::zeros(d_numOfExamples, d_numOfFeatures, CV_64F);
			m_dataClass = Mat::ones(d_numOfExamples, 2, CV_64F)*(-1.0);
			m_vectorsOfCenters = Models(d_numOfClusters, d_numOfFeatures);
			for (int i = 0; i < d_numOfExamples; i++)
				for (int j = 0; j < d_numOfFeatures; j++) {
					if ((c = inputFile.peek()) != '\n'&&!inputFile.eof())
						inputFile >> m_dataSet.at<double>(i, j);
					else {
						isWrong = true;
						break;
					}
					if (isWrong)
						break;
				}
		}
		inputFile.close();
	}
}

Mat kMeansFilters::preProcessing() {
	//预处理
	if (!isInited)
		return m_dataSet;
	m_minVal = Mat::zeros(1, d_numOfFeatures, CV_64F);
	m_maxVal = Mat::zeros(1, d_numOfFeatures, CV_64F);
	Mat m_sub = Mat::zeros(1, d_numOfFeatures, CV_64F);
	for (int i = 0; i < d_numOfFeatures; i++) {
		minMaxIdx(m_dataSet.col(i), &m_minVal.at<double>(0, i), &m_maxVal.at<double>(0, i));
	}
	for (int i = 1; i < d_numOfExamples; i++) {
		m_dataSet.row(i) = (m_dataSet.row(i) - m_minVal) / (m_maxVal - m_minVal);
	}
	
	return m_dataSet;
}

Models kMeansFilters::createInitVector() {
	if (!isInited)
		return m_vectorsOfCenters;
	
	//随机生成
	srand((unsigned)time(NULL));
	Mat randomVectorIndex = Mat::zeros(1, d_numOfClusters, CV_8U);
	for (int i = 0; i < d_numOfClusters; i++) {
		int index = rand() % (d_numOfExamples/d_numOfClusters*(i+1) - d_numOfExamples / d_numOfClusters*(i))+ d_numOfExamples / d_numOfClusters*(i);
		randomVectorIndex.at<uchar>(0, i) = index;
		m_vectorsOfCenters.modelUpdate(i, m_dataSet.row(index));
	}
	cout << randomVectorIndex.row(0) << endl;
	return m_vectorsOfCenters;
}
Mat kMeansFilters::getCluster(vector<int> indexs) {
	int size = indexs.size();
	Mat tmp = Mat::zeros(size, d_numOfFeatures, CV_64F);
	for (int i = 0; i < size; i++) {
		int index = indexs.at(i);
		m_dataSet.row(index).copyTo(tmp.row(i));
	}
	return tmp;
}


Mat kMeansFilters::meanOfCluster(vector<int> indexs) {
	int size = indexs.size();
	Mat tmp = getCluster(indexs);
	Mat result = Mat::zeros(1, d_numOfFeatures, CV_64F);
	for (int i = 0; i < d_numOfFeatures; i++) {
		result.at<double>(0, i) = mean(tmp.col(i))[0];
	}
	
	return result;
}

Models kMeansFilters::trainClusters(int maxTimes) {
	//训练顺序，样本集生成完毕后
	//数据预处理+生成初始向量+（簇分配+中心迁移）*n
	if (!isInited)
		return m_vectorsOfCenters;
	int counts = 0;
	bool isTrained = false;

	m_dataSet = preProcessing();
	m_vectorsOfCenters = createInitVector();
	while (counts < maxTimes&&!isTrained) {
		isTrained = true;
		//簇分配
		for (int i = 0; i < d_numOfExamples; i++) {
			pair<int, double> tmpCluster = m_vectorsOfCenters.matchModel(m_dataSet.row(i));
			m_dataClass.at<double>(i, 1) = tmpCluster.second;
			if ((int)m_dataClass.at<double>(i, 0) != tmpCluster.first)
				isTrained = false;
			m_dataClass.at<double>(i, 0) = (double)tmpCluster.first;
		}

		//中心迁移
		vector<vector<int>> indexsOfClusters = vector<vector<int>>();
		for (int i = 0; i < d_numOfClusters; i++) {
			indexsOfClusters.push_back(vector<int>());
		}
		for (int i = 0; i < d_numOfExamples; i++) {
			int k = (int)m_dataClass.at<double>(i, 0);
			indexsOfClusters.at(k).push_back(i);
		}
		for (int i = 0; i < d_numOfClusters; i++) {
			Mat tmpFeatures = Mat::zeros(1, d_numOfFeatures, CV_64F);
			Mat tmpCluster = getCluster(indexsOfClusters.at(i));
			tmpFeatures = meanOfCluster(indexsOfClusters.at(i));
			m_vectorsOfCenters.modelUpdate(i, tmpFeatures);
			m_vectorsOfCenters.setThreshold(i, tmpCluster);
		}
	}
	cout << m_dataClass.col(0) << endl;
	return m_vectorsOfCenters;
}


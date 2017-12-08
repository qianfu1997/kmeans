#include"KMeans.h"
#include"Models.h"
using namespace std;
using namespace cv;

//以下内容为离散特征值选择与取值范围
//与前科嫌疑人员同时段出入同一网吧
//取值范围0-2,0表示不存在；1表示一小时内；2表示十分钟内
#define FeatureA 2.0
//在多县出现，取值0-10
#define FeatureB 10.0
//在多市出现 取值0-10
#define FeatureC 10.0
//在多省出现 取值0-10
#define FeatureD 10.0
//第1、3类人员出入境次数，取值范围0-100
#define FeatureE 100.0
//与前科嫌疑人员入住同一宾馆的，取值范围0-2
#define FeatureF 2.0
//每十日内在本市不同九点旅馆登记次数 取值范围0-100
#define FeatureG 100.0
//1-6时登记离住宿次数 取值范围0-50
#define FeatureH 50.0
//1-6时出入网吧次数 取值范围0-50
#define FeatureI 50.0
//典当旧货行为次数 取值范围0-100
#define FeatureJ 100.0

//定义样本集数量
#define exampleSize 100
//定义测试集数量
#define testSize 10
//定义簇数量
#define clusterNum 4
//定义最大循环次数
#define trainTimes 10000
//定义特征数量(请配合上方定义修改)
#define featureNums 10

//生成随机样本集
Mat createExamples(int nums);

int main(void) {
	ifstream modelFile("model.txt");
	ofstream newModelFile;
	char c;
	int kind;
	Models test_model;
	if (modelFile.is_open()) {
		if ((c = modelFile.peek()) != '\n'&&!modelFile.eof()) {
			//存在已经训练好的模型
			cout << "1 训练新模型；2 利用上次生成的模型进行预测：" << endl;
			cin >> kind;
			modelFile.close();
			if (kind == 1) {
				Mat dataset = createExamples(exampleSize);
				kMeansFilters test_filter(dataset, clusterNum);
				test_model = test_filter.trainClusters(trainTimes);
				test_model.outputModels();
				cout << "done!" << endl;
			}
			else {
				test_model = Models("model.txt");
				Mat testset = createExamples(testSize);
				vector<int> result = vector<int>(testSize);
				for (int i = 0; i < testSize; i++) {
					pair<int, double> tmp = test_model.matchModel(testset.row(i));
					result.push_back(tmp.first);
				}
				for (int i = 0; i < result.size(); i++)
					cout << result.at(i) << "\t";
				cout << "done!" << endl;
			}
		}
		else {
			cout << "暂无模型，将生成新模型。" << endl;
			modelFile.close();
			Mat dataset = createExamples(exampleSize);
			cout << dataset.row(0) << endl;
			kMeansFilters test_filter(dataset, clusterNum);
			test_model = test_filter.trainClusters(trainTimes);
			test_model.outputModels();
			cout << "done!" << endl;
		}
	}
}

Mat createExamples(int nums) {
	Mat dataset = Mat::zeros(nums, featureNums, CV_64F);
	srand((unsigned)time(NULL));
	//填充
	for (int i = 0; i < nums; i++) {
		dataset.at<double>(i, 0) = rand() / double(RAND_MAX)*FeatureA;
		dataset.at<double>(i, 1) = rand() / double(RAND_MAX)*FeatureB;
		dataset.at<double>(i, 2) = rand() / double(RAND_MAX)*FeatureC;
		dataset.at<double>(i, 3) = rand() / double(RAND_MAX)*FeatureD;
		dataset.at<double>(i, 4) = rand() / double(RAND_MAX)*FeatureE;
		dataset.at<double>(i, 5) = rand() / double(RAND_MAX)*FeatureF;
		dataset.at<double>(i, 6) = rand() / double(RAND_MAX)*FeatureG;
		dataset.at<double>(i, 7) = rand() / double(RAND_MAX)*FeatureH;
		dataset.at<double>(i, 8) = rand() / double(RAND_MAX)*FeatureI;
		dataset.at<double>(i, 9) = rand() / double(RAND_MAX)*FeatureJ;
	}
	return dataset;
}
#include"KMeans.h"
#include"Models.h"
using namespace std;
using namespace cv;

//��������Ϊ��ɢ����ֵѡ����ȡֵ��Χ
//��ǰ��������Աͬʱ�γ���ͬһ����
//ȡֵ��Χ0-2,0��ʾ�����ڣ�1��ʾһСʱ�ڣ�2��ʾʮ������
#define FeatureA 2.0
//�ڶ��س��֣�ȡֵ0-10
#define FeatureB 10.0
//�ڶ��г��� ȡֵ0-10
#define FeatureC 10.0
//�ڶ�ʡ���� ȡֵ0-10
#define FeatureD 10.0
//��1��3����Ա���뾳������ȡֵ��Χ0-100
#define FeatureE 100.0
//��ǰ��������Ա��סͬһ���ݵģ�ȡֵ��Χ0-2
#define FeatureF 2.0
//ÿʮ�����ڱ��в�ͬ�ŵ��ùݵǼǴ��� ȡֵ��Χ0-100
#define FeatureG 100.0
//1-6ʱ�Ǽ���ס�޴��� ȡֵ��Χ0-50
#define FeatureH 50.0
//1-6ʱ�������ɴ��� ȡֵ��Χ0-50
#define FeatureI 50.0
//�䵱�ɻ���Ϊ���� ȡֵ��Χ0-100
#define FeatureJ 100.0

//��������������
#define exampleSize 100
//������Լ�����
#define testSize 10
//���������
#define clusterNum 4
//�������ѭ������
#define trainTimes 10000
//������������(������Ϸ������޸�)
#define featureNums 10

//�������������
Mat createExamples(int nums);

int main(void) {
	ifstream modelFile("model.txt");
	ofstream newModelFile;
	char c;
	int kind;
	Models test_model;
	if (modelFile.is_open()) {
		if ((c = modelFile.peek()) != '\n'&&!modelFile.eof()) {
			//�����Ѿ�ѵ���õ�ģ��
			cout << "1 ѵ����ģ�ͣ�2 �����ϴ����ɵ�ģ�ͽ���Ԥ�⣺" << endl;
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
			cout << "����ģ�ͣ���������ģ�͡�" << endl;
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
	//���
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
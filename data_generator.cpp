#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <fstream>
#include <vector>
#include <stdlib.h>
#include <time.h>
#include <array>
#include <typeinfo>
using namespace std;


int QTL_1, QTL_2, QTL_3, QTL_4;
int Operation_Skip;
int main() {
	ofstream fout;
	srand(time(NULL));
	vector<int>Stage;
	vector<int>Job;
	vector<int>Operation;
	vector<int>A_machine;
	int Queue_time;

	int Low_L_Rework = 5;
	int Low_H_Rework = 20;

	int High_L_Rework = 80;
	int High_H_Rework = 100;

	int Low_queue_time = 5;
	int High_queue_time = 20;
	char filename[50];
	int num;
	int due_date_k;

	Job = { 30,50,70 };
	Stage = { 5,10 };
	A_machine = { 3, 5, 7 };
	Queue_time = 2;


	for (int k = 0; k < Job.size(); k++) {

		for (int y = 0; y < Stage.size(); y++) {
			for (int r = 0; r < A_machine.size(); r++) {
				for (num = 0; num < 10; num++) {
					sprintf(filename, "C:/Users/user/source/repos/my_thesis/Data/TData_%d_%d_%d_%d.txt", Job[k], Stage[y], A_machine[r], num);

					fout.open(filename);

					fout << Job[k] << endl;
					fout << Stage[y] << endl;
					fout << A_machine[r] << endl;
					fout << Queue_time << endl;
					due_date_k = A_machine[r];



					int a = Stage[y] / 2;

					QTL_1 = rand() % a + 1;
					QTL_2 = rand() % a + 1;

					while (QTL_1 == QTL_2) {

						QTL_2 = rand() % a + 1;
					}

					int temp;
					if (QTL_1 > QTL_2) {
						temp = QTL_1;
						QTL_1 = QTL_2;
						QTL_2 = temp;
					}
					QTL_3 = rand() % a + 1;
					QTL_4 = rand() % a + 1;

					QTL_3 += a;
					QTL_4 += a;



					while (QTL_3 == QTL_4) {

						QTL_4 = rand() % a + 1;
						QTL_4 += a;
					}


					if (QTL_3 > QTL_4) {
						temp = QTL_3;
						QTL_3 = QTL_4;
						QTL_4 = temp;
					}

					fout << endl;
					
					for (int i = 0; i < Job[k]; i++) {
						int sum = 0;
						fout << rand() % (Low_H_Rework - Low_L_Rework) + Low_L_Rework << " ";
						fout << rand() % (High_H_Rework - High_L_Rework) + High_L_Rework << endl;
						fout << rand() % (High_queue_time - Low_queue_time) + Low_queue_time << " ";	//QTL1

						//54 - 98 : QTL 범위, Operation Skipping 정하기

						fout << QTL_1 << " ";
						fout << QTL_2 << " ";

						Operation_Skip = rand() % ((QTL_2 - 1) - QTL_1 + 1) + QTL_1;
						//fout << Operation_Skip << endl;
						fout << endl;



						fout << rand() % (High_queue_time - Low_queue_time) + Low_queue_time << " ";	//QTL2


						fout << QTL_3 << " ";
						fout << QTL_4 << " ";
						Operation_Skip = rand() % ((QTL_4 - 1) - QTL_3 + 1) + QTL_3;
						//fout << Operation_Skip << endl;
						fout << endl;
						for (int x = 0; x < Stage[y]; x++) {

							int Processing_time = rand() % (100 - 5) + 5;
							if (x == Stage[y] - 1) {
								fout << Processing_time;
								sum += Processing_time;

							}
							else {
								fout << Processing_time << " ";
								sum += Processing_time;

							}
						}
						




						fout << endl;
						float TF, RDD;

						TF = rand() % 3 + 7;
						TF = TF / 10;

						RDD = rand() % 5 + 4;
						RDD = RDD / 10;

						int T = sum * (1 - TF + (RDD / 2)) / due_date_k;

						int N = sum * (1 - TF - (RDD / 2)) / due_date_k;
						

						if (N <= 0) {
							N = sum;
						}
						
						int due_date = rand() % (T - N + 1) + N;

						if (i != Job[k] - 1) {
							fout << due_date << "\n";


							fout << endl;

						}
						else {
							fout << due_date;


						}
						
					}
					fout.close();
				}
			}
		}

	}


}
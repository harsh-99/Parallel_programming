//Just enter the string of Char in Capslock, the code itself changes the number of thread 1,2,4. and average value of time for 
//making database, calculating redundancy and searching is stored in a text file named out.txt
//to compile use g++ -fopenmp search.cpp

#include <iostream>
#include <stdlib.h>
#include <omp.h>
#include<bits/stdc++.h>
#include <cmath>
#include <fstream>

using namespace std;

// Declare structure to return two value from a function 
struct ret{
	float a;
	float b;
};

// used to create the data of given length and using char {A, G, T, C}
float form_data(char** data, int len, int str_len, int num_threads){
	float wtime;
	char a[] = {'A', 'C', 'T', 'G'};
	wtime = omp_get_wtime();
	for(int i=0;i<len;i++){
		for(int j=0;j<str_len;j++){
			data[i][j] = a[rand()%4];
		}
	}
	return omp_get_wtime()- wtime;
}

//Calculating redundancy using the hashing method in order to reduce and time and get an approximate value of redundany 
ret redundancy(char** data, int len, int str_len, int num_threads)
{
	long max_val = 0;
	max_val = (int)10*len;
	ret variable;
	int* hash = new int[max_val]();
	int c = 0;
	float redun;
	long val = 0;
	double wtime = omp_get_wtime();
	#pragma omp parallel num_threads(num_threads)
	{
		#pragma omp for private(val)
		for(int i = 0; i < len; i++)
		{
			val = 0;
			for(int j = 0; j < str_len; j++)
			{
				if(data[i][j] == 'A')
					val += 0*pow(4, j);
				else if(data[i][j] == 'C')
					val += 1*pow(4, j);
				else if(data[i][j] == 'T')
					val += 2*pow(4, j);
				else if(data[i][j] == 'G')
					val += 3*pow(4, j);
			}
			val = val%max_val;
			if(hash[val] == 1)
				continue;
			#pragma omp critical
			{
				hash[val] = 1;
				c++;
			}
		}
	}
	wtime = omp_get_wtime()-wtime;
	// cout<<"The time taken for calculating redundancy is "<<wtime<<"s\n";
	// cout<<"Hashing is used to find the redundancy\n";  //the result calculated will be more then the actual 
	// cout<<"Calculated Redundancy = "<<(len - c)*100.0/len<<" %"<<endl;
	variable.a = (len - c)*100.0/len;
	variable.b = wtime;
	return variable;
}



int main(){
	int length = 6400000;
	int str_len = 15;
	char a[str_len];
	cout<<"Enter the string of length 15 which you need to find\n";
	for(int i=0;i<str_len;i++){
		cin>>a[i];
	}
	int n_threads[] = {1,2,4};
	ofstream myfile ("out.txt");
	
	for(int l=0; l<3;l++){								//this is used for getting different number of threads

		cout<<"Using "<<n_threads[l]<<" number of thread\n";
		float data_time;
		char** data = new char* [length];
		for(int i=0;i<length;i++){
			data[i] = new char [str_len];
		}

		data_time = form_data(data, length, str_len, n_threads[l]);

		float sum_redun, sum_wtime, sum_wtime_r;
		double wtime[10];
		float redun[10];
		sum_redun = sum_wtime = sum_wtime_r =0;

		for(int t=0;t<10;t++){							//getting an average over 10 iteration as instructed in assignment
			if(t>0) cout<<"Entering new iteration using "<<n_threads[l]<<" threads"<<endl;
			int i, j, flag;
			ret variable;
			vector<int> index;
			wtime[t] = omp_get_wtime();
			#pragma omp parallel num_threads(n_threads[l])
			{
				#pragma omp for private(flag, j) schedule(static)
					for(i=0;i<length;i++){
						flag = 0;
						for(j=0;j<15;j++){
							if(data[i][j] != a[j]) {flag = 1; break;}
						}
						if(flag == 0) index.push_back(i);
					}
			}
			wtime[t] = 	omp_get_wtime()-wtime[t];
			// cout<<"The time taken is "<<wtime[t]<<"s\n";
			cout<<"Count = "<<index.size()<<"and it's index is\n";
			for(int i=0; i<index.size(); i++){
				cout<<index[i]<<"  ";
				// cout<<"hello";
				// index.pop_back();
			}
			cout<<"\n";
			variable = redundancy(data, length, 15, n_threads[l]);
			sum_wtime += wtime[t];
			sum_redun +=variable.a; 
			sum_wtime_r +=variable.b;
		}
		cout<<"the average redundancy is "<<sum_redun/10<<" \n";
		cout<<"the average time for searching is "<<sum_wtime/10<<"s \n";
		cout<<"the average time for calculating redundancy is "<<sum_wtime_r/10<<"s \n";
		myfile<<"Using number of thread as "<<n_threads[l]<<" Writing the value of average time for searching, calculating redundancy and data creation respectively "<<sum_wtime/10<<"s and "<<sum_wtime_r/10<<"s and "<<data_time<<"s. \n";
		}

		myfile.close();

}
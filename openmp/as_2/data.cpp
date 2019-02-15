#include <iostream>
#include <stdlib.h>
#include <fstream>

using namespace std;

int main(){
	char a[] = {'A', 'C', 'T', 'G'};
	int length = 6400000;
	ofstream myfile ("example.txt");
	char** data = new char* [length];

	for(int i=0;i<length;i++){
		data[i] = new char [15];
	}
	for(int i=0;i<length;i++){
		for(int j=0;j<15;j++){
			data[i][j] = a[rand()%4];
		}
	}
	if (myfile.is_open())
	  {
	    for(int count = 0; count < length; count ++){
	    	for(int j=0;j<15;j++){
	        	myfile << data[count][j];
	    	}
	    	myfile<<"\n";
	    }

	    myfile.close();
	  }


}